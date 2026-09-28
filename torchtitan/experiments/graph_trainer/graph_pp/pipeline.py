# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import copy
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    BACKWARD_WEIGHT,
    FULL_BACKWARD,
    get_schedule_class,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.pipeline_parallel import (
    _build_get_mesh_callback,
    _build_pipeline_schedule,
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
    _split_module,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.graph_builder import (
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD,
    BACKWARD_WEIGHT_WITH_REDUCE_GRAD,
    BACKWARD_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_NOGRADACCUM,
    FORWARD_BACKWARD_WITH_REDUCE_GRAD,
    FORWARD_BACKWARD_WITH_UNSHARD,
    GraphRuntime,
    joint_forward_backward_computation_type,
    register_graph_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.protocols.model import BaseModel


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRuntimeFSDPPolicy:
    """Place each FSDP boundary in a schedule action or an edge microbatch."""

    schedule_fsdp_param_unshard: bool
    schedule_fsdp_grad_reduction: bool
    unshard_in_first_microbatch: bool = False
    reduce_grad_in_last_microbatch: bool = False

    @property
    def split_fsdp_param_unshard(self) -> bool:
        return self.schedule_fsdp_param_unshard or self.unshard_in_first_microbatch

    @property
    def split_fsdp_grad_reduction(self) -> bool:
        return self.schedule_fsdp_grad_reduction or self.reduce_grad_in_last_microbatch


@dataclass(frozen=True)
class GraphRuntimeGradientAccumulationPolicy:
    accumulate_in_graph: bool
    fuse_wgrad_accumulation: bool


def resolve_graph_runtime_fsdp_policy(
    compile_config: GraphTrainerCompileConfig,
    *,
    num_microbatches: int,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> GraphRuntimeFSDPPolicy:
    """Resolve topology-dependent FSDP graph boundaries.

    PP=1 keeps FSDP operations in the joint graph for one microbatch and
    extracts them for gradient accumulation. Edge-microbatch modes retain the
    extracted boundary in only the first or last joint graph. PP>1 extracts
    both boundaries by default.
    """
    if not fsdp_enabled:
        if compile_config.fsdp_param_unshard_mode not in ("auto", "in_graph"):
            raise ValueError("Selected FSDP parameter unsharding mode requires FSDP")
        if compile_config.fsdp_gradient_sync_mode not in ("auto", "in_graph"):
            raise ValueError("Selected FSDP gradient reduction mode requires FSDP")
        return GraphRuntimeFSDPPolicy(
            schedule_fsdp_param_unshard=False,
            schedule_fsdp_grad_reduction=False,
        )

    if not pp_enabled:
        has_gradient_accumulation = num_microbatches > 1
        unshard_in_first_microbatch = (
            compile_config.fsdp_param_unshard_mode == "in_graph_only_first_mb"
        )
        reduce_grad_in_last_microbatch = (
            compile_config.fsdp_gradient_sync_mode == "in_graph_only_last_mb"
        )
        if (
            unshard_in_first_microbatch or reduce_grad_in_last_microbatch
        ) and not has_gradient_accumulation:
            raise ValueError(
                "First/last-microbatch FSDP modes require gradient accumulation"
            )
        if (
            unshard_in_first_microbatch
            and compile_config.fsdp_gradient_sync_mode == "in_graph"
        ) or (
            reduce_grad_in_last_microbatch
            and compile_config.fsdp_param_unshard_mode == "in_graph"
        ):
            raise ValueError(
                "First/last-microbatch FSDP boundaries cannot be combined with "
                "the other boundary inside every joint graph"
            )
        return GraphRuntimeFSDPPolicy(
            schedule_fsdp_param_unshard=(
                compile_config.fsdp_param_unshard_mode == "extracted_in_schedule_stage"
                or (
                    compile_config.fsdp_param_unshard_mode == "auto"
                    and has_gradient_accumulation
                )
            ),
            schedule_fsdp_grad_reduction=(
                compile_config.fsdp_gradient_sync_mode == "deferred_as_schedule_stage"
                or (
                    compile_config.fsdp_gradient_sync_mode == "auto"
                    and has_gradient_accumulation
                )
            ),
            unshard_in_first_microbatch=unshard_in_first_microbatch,
            reduce_grad_in_last_microbatch=reduce_grad_in_last_microbatch,
        )

    if compile_config.fsdp_param_unshard_mode in (
        "in_graph",
        "in_graph_only_first_mb",
    ):
        raise ValueError("PP>1 GraphPP requires extracted FSDP parameter unsharding")
    if compile_config.fsdp_gradient_sync_mode in (
        "in_graph",
        "in_graph_only_last_mb",
    ):
        raise ValueError("PP>1 GraphPP requires deferred FSDP gradient reduction")
    return GraphRuntimeFSDPPolicy(
        schedule_fsdp_param_unshard=True,
        schedule_fsdp_grad_reduction=True,
    )


def resolve_graph_runtime_gradient_accumulation_policy(
    compile_config: GraphTrainerCompileConfig,
    *,
    num_microbatches: int,
    pp_enabled: bool,
    fsdp_enabled: bool,
    split_fsdp_grad_reduction: bool,
) -> GraphRuntimeGradientAccumulationPolicy:
    """Resolve gradient accumulation placement and optional WGrad fusion."""
    accumulation_mode = compile_config.gradient_accumulation_mode
    fusion_mode = compile_config.gradient_accum_in_wgrad_fusion
    if accumulation_mode == "runtime" and fusion_mode == "enabled":
        raise ValueError(
            "WGrad accumulation fusion requires in-graph gradient accumulation"
        )
    if pp_enabled:
        if accumulation_mode == "in_graph" or fusion_mode == "enabled":
            raise ValueError(
                "PP>1 requires runtime gradient accumulation until its schedule "
                "spans the complete optimizer step"
            )
        return GraphRuntimeGradientAccumulationPolicy(
            accumulate_in_graph=False,
            fuse_wgrad_accumulation=False,
        )
    if fusion_mode == "enabled":
        if not compile_config.enable_passes:
            raise ValueError("WGrad accumulation fusion requires graph passes")
        if "fuse_wgrad_accumulation_pass" in compile_config.disable_passes:
            raise ValueError(
                "WGrad accumulation fusion is enabled but its pass is disabled"
            )
        if fsdp_enabled and not split_fsdp_grad_reduction:
            raise ValueError(
                "WGrad accumulation fusion with FSDP requires "
                "FSDP gradient reduction outside the reusable graph"
            )

    auto_accumulate_in_graph = num_microbatches > 1
    accumulate_in_graph = accumulation_mode == "in_graph" or (
        accumulation_mode == "auto"
        and (auto_accumulate_in_graph or fusion_mode == "enabled")
    )
    can_fuse_wgrad = not fsdp_enabled or split_fsdp_grad_reduction
    fuse_wgrad_accumulation = (
        accumulate_in_graph
        and can_fuse_wgrad
        and compile_config.enable_passes
        and "fuse_wgrad_accumulation_pass" not in compile_config.disable_passes
        and (
            fusion_mode == "enabled"
            or (fusion_mode == "auto" and compile_config.numerics_changing_optim)
        )
    )
    return GraphRuntimeGradientAccumulationPolicy(
        accumulate_in_graph=accumulate_in_graph,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )


def _validate_spmd_graph_runtime_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphRuntime requires --compile.mode aot_fx_trace")


def _new_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    loss_fn: LossFunction,
) -> _PipelineScheduleRuntime:
    """Create an empty runtime schedule for one SPMD model stage."""

    def scalar_loss_fn(*args: object, **kwargs: object) -> torch.Tensor:
        loss = loss_fn(*args, **kwargs)
        return loss[0] if isinstance(loss, tuple) else loss

    return _PipelineScheduleRuntime(
        [stage],
        n_microbatches=num_microbatches,
        loss_fn=scalar_loss_fn,
        scale_grads=False,
        backward_requires_autograd=False,
    )


def _make_spmd_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    fsdp_enabled: bool,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    accumulate_gradients_in_graph: bool,
) -> _PipelineScheduleRuntime:
    """Build an SPMD schedule with joint train-step actions."""
    schedule = _new_spmd_runtime_schedule(
        stage,
        num_microbatches=num_microbatches,
        loss_fn=loss_fn,
    )
    fsdp_reshard_after_forward = (
        get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        if fsdp_enabled
        else None
    )
    if (
        fsdp_policy.unshard_in_first_microbatch
        and fsdp_reshard_after_forward is not False
    ):
        raise ValueError(
            "in_graph_only_first_mb requires FSDP parameters to remain "
            "unsharded across microbatches"
        )
    reuse_unsharded_parameters = (
        fsdp_policy.split_fsdp_param_unshard and fsdp_reshard_after_forward is False
    )
    reusable_computation_type = joint_forward_backward_computation_type(
        split_fsdp_param_unshard=fsdp_policy.split_fsdp_param_unshard,
        split_fsdp_grad_reduction=fsdp_policy.split_fsdp_grad_reduction,
    )
    actions: list[_Action] = []
    if reuse_unsharded_parameters and fsdp_policy.schedule_fsdp_param_unshard:
        actions.append(_Action(0, UNSHARD))
    for microbatch_index in range(num_microbatches):
        if fsdp_policy.schedule_fsdp_param_unshard and not reuse_unsharded_parameters:
            actions.append(_Action(0, UNSHARD))
        computation_type = reusable_computation_type
        if microbatch_index == 0 and fsdp_policy.unshard_in_first_microbatch:
            computation_type = FORWARD_BACKWARD_WITH_UNSHARD
        elif microbatch_index == 0 and accumulate_gradients_in_graph:
            computation_type = FORWARD_BACKWARD_NOGRADACCUM
        elif (
            microbatch_index == num_microbatches - 1
            and fsdp_policy.reduce_grad_in_last_microbatch
        ):
            computation_type = FORWARD_BACKWARD_WITH_REDUCE_GRAD
        actions.append(_Action(0, cast(Any, computation_type), microbatch_index))
        if fsdp_policy.schedule_fsdp_param_unshard and not reuse_unsharded_parameters:
            actions.append(_Action(0, RESHARD))
    if fsdp_policy.schedule_fsdp_grad_reduction:
        actions.append(_Action(0, REDUCE_GRAD))
    if reuse_unsharded_parameters:
        actions.append(_Action(0, RESHARD))
    # Upstream schedule validation only recognizes separate F/B actions. PP=1
    # has no pipeline communication to lower, so install the already-lowered
    # joint schedule directly instead of misrepresenting FORWARD_BACKWARD
    # as a container of split actions.
    schedule.stage_index_to_group_rank = {0: 0}
    stage.stage_index_to_group_rank = schedule.stage_index_to_group_rank
    schedule.pipeline_order_with_comms = {0: actions}
    return schedule


def _set_graph_backward_actions(
    schedule: _PipelineScheduleRuntime,
    *,
    extract_fsdp_grad_reduction: bool,
) -> None:
    """Replace PyTorch backward actions with explicit GraphRuntime variants."""

    def replace_action(action: _Action) -> _Action:
        computation_type = action.computation_type
        if computation_type == FULL_BACKWARD:
            computation_type = (
                BACKWARD if extract_fsdp_grad_reduction else BACKWARD_WITH_REDUCE_GRAD
            )
        elif computation_type == BACKWARD_WEIGHT and not extract_fsdp_grad_reduction:
            computation_type = BACKWARD_WEIGHT_WITH_REDUCE_GRAD

        sub_actions = (
            None
            if action.sub_actions is None
            else tuple(replace_action(sub_action) for sub_action in action.sub_actions)
        )
        return _Action(
            action.stage_index,
            cast(Any, computation_type),
            action.microbatch_index,
            sub_actions,
        )

    schedule.pipeline_order_with_comms = {
        rank: [replace_action(action) for action in actions]
        for rank, actions in schedule.pipeline_order_with_comms.items()
    }


def _make_pipeline_parallel_runtime_schedule(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    extract_fsdp_grad_reduction: bool,
) -> tuple[_PipelineScheduleRuntime, _PipelineScheduleRuntime]:
    """Build a real-PP schedule through the upstream schedule implementation."""
    schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=num_microbatches,
        stages=stages,  # pyrefly: ignore [bad-argument-type]
        loss_fn=loss_fn,
        backward_requires_autograd=False,
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)
    activation_liveness_schedule = copy(schedule)
    activation_liveness_schedule.pipeline_order_with_comms = {
        rank: list(actions)
        for rank, actions in schedule.pipeline_order_with_comms.items()
    }
    _set_graph_backward_actions(
        schedule,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
    )
    return schedule, activation_liveness_schedule


def _validate_graph_pp_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPP requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPP does not support --compile.precompile_artifact_dir yet. "
            "Trace and graph construction are stage-local runtime operations."
        )
    if parallelism.fsdp_reshard_after_forward == "always":
        raise ValueError(
            "GraphPP assumes ZeRO-2 style FSDP with "
            "--parallelism.fsdp_reshard_after_forward default/never, not always."
        )
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if not issubclass(schedule_class, _PipelineScheduleRuntime):
        raise ValueError(
            "GraphPP currently requires a runtime PP schedule such as "
            "Interleaved1F1B, ZBVZeroBubble, or DualPipeV. "
            f"Got {parallelism.pipeline_parallel_schedule}."
        )


def _register_graph_runtime(
    schedule: _PipelineScheduleRuntime,
    *,
    activation_liveness_schedule: _PipelineScheduleRuntime | None = None,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    gradient_accumulation_policy: GraphRuntimeGradientAccumulationPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    trainer_config: "GraphTrainer.Config | None",
    parallel_dims: ParallelDims,
    warn_if_cuda_graph_pass_requested: bool,
) -> GraphRuntime:
    """Bind GraphTrainer graph construction to an already chosen schedule.

    Args:
        trainer_config: Full Trainer configuration for PP=1, or ``None`` for
            PP>1 schedules.
    """
    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=fsdp_policy.schedule_fsdp_param_unshard,
        extract_fsdp_grad_reduction=fsdp_policy.schedule_fsdp_grad_reduction,
        unshard_in_first_microbatch=fsdp_policy.unshard_in_first_microbatch,
        reduce_grad_in_last_microbatch=fsdp_policy.reduce_grad_in_last_microbatch,
        accumulate_gradients_in_graph=(
            gradient_accumulation_policy.accumulate_in_graph
        ),
        fuse_wgrad_accumulation=(gradient_accumulation_policy.fuse_wgrad_accumulation),
        trainer_config=trainer_config,
        parallel_dims=parallel_dims,
    )
    if warn_if_cuda_graph_pass_requested:
        graph_provider._warn_if_cuda_graph_pass_requested()
    return register_graph_schedule(
        schedule,
        graph_provider=graph_provider,
        activation_liveness_schedule=activation_liveness_schedule,
        unshard_in_first_microbatch=fsdp_policy.unshard_in_first_microbatch,
        reduce_grad_in_last_microbatch=fsdp_policy.reduce_grad_in_last_microbatch,
    )


def _make_spmd_graph_runtime(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    gradient_accumulation_policy: GraphRuntimeGradientAccumulationPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    trainer_config: "GraphTrainer.Config",
    parallel_dims: ParallelDims,
) -> GraphRuntime:
    """Build SPMD execution with joint microbatch and optional FSDP actions."""
    if (
        fsdp_policy.unshard_in_first_microbatch
        or fsdp_policy.reduce_grad_in_last_microbatch
    ) and not gradient_accumulation_policy.accumulate_in_graph:
        raise ValueError(
            "First/last-microbatch FSDP graphs require in-graph gradient "
            "accumulation"
        )
    requires_graph_extraction = (
        fsdp_policy.split_fsdp_param_unshard
        or fsdp_policy.split_fsdp_grad_reduction
        or gradient_accumulation_policy.accumulate_in_graph
    )
    if requires_graph_extraction and compile_config.ep_overlap.enabled:
        raise ValueError(
            "GraphRuntime scheduled SPMD graph extraction does not support "
            "--compile.ep_overlap.enabled yet. The EP-overlap graph rewrites "
            "have not been validated with extracted runtime callables."
        )
    if requires_graph_extraction and compile_config.memory_policy == "sac_and_offload":
        raise ValueError(
            "GraphRuntime scheduled SPMD graph extraction does not support "
            "--compile.memory_policy sac_and_offload yet. Graph extraction "
            "must preserve offload and reload pairs."
        )
    if (
        requires_graph_extraction
        and compile_config.pass_pipeline in PASS_PIPELINE_REGISTRY
    ):
        raise ValueError(
            "GraphRuntime scheduled SPMD graph extraction does not support "
            "custom pass pipelines yet"
        )
    trace_preparer_names = set(trace_input_preparer_keys(compile_config))
    unsupported_preparers = trace_preparer_names.intersection(
        TRACE_INPUT_PREPARERS.keys() | TRACE_CALL_INPUT_PREPARERS.keys()
    )
    if requires_graph_extraction and unsupported_preparers:
        raise ValueError(
            "GraphRuntime scheduled SPMD graph extraction does not support "
            "trace-input preparers "
            f"yet: {sorted(unsupported_preparers)}"
        )
    schedule = _make_spmd_runtime_schedule(
        stage,
        num_microbatches=num_microbatches,
        parallelism=parallelism,
        loss_fn=loss_fn,
        fsdp_enabled=parallel_dims.fsdp_enabled,
        fsdp_policy=fsdp_policy,
        accumulate_gradients_in_graph=(
            gradient_accumulation_policy.accumulate_in_graph
        ),
    )
    return _register_graph_runtime(
        schedule,
        fsdp_policy=fsdp_policy,
        gradient_accumulation_policy=gradient_accumulation_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        trainer_config=trainer_config,
        parallel_dims=parallel_dims,
        warn_if_cuda_graph_pass_requested=False,
    )


def _make_pipeline_parallel_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    fsdp_policy: GraphRuntimeFSDPPolicy,
    gradient_accumulation_policy: GraphRuntimeGradientAccumulationPolicy,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    parallel_dims: ParallelDims,
) -> GraphRuntime:
    """Build graph execution around a real pipeline-parallel schedule."""
    schedule, activation_liveness_schedule = _make_pipeline_parallel_runtime_schedule(
        stages,
        num_microbatches=num_microbatches,
        parallelism=parallelism,
        loss_fn=loss_fn,
        extract_fsdp_grad_reduction=fsdp_policy.schedule_fsdp_grad_reduction,
    )
    return _register_graph_runtime(
        schedule,
        activation_liveness_schedule=activation_liveness_schedule,
        fsdp_policy=fsdp_policy,
        gradient_accumulation_policy=gradient_accumulation_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        trainer_config=None,
        parallel_dims=parallel_dims,
        warn_if_cuda_graph_pass_requested=True,
    )


def make_graph_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    trainer_config: "GraphTrainer.Config | None",
) -> GraphRuntime:
    """Build the schedule, stage graphs, and runtime for GraphTrainer.

    Descriptive notation
    --------------------
    ``s`` is a stage index and ``m`` is a microbatch index:

    - ``FORWARD_BACKWARD_NOGRADACCUM(s, 0)`` runs the first PP=1 joint graph
      without gradient accumulation and returns unsharded gradients that become
      gradient accumulators for further graphs.
    - ``FORWARD_BACKWARD(s, m)`` runs a repeated PP=1 joint graph without
      UNSHARD or REDUCE_GRAD, takes gradient accumulator inputs, and accumulates
      unsharded gradients into them in place.
    - ``FORWARD_BACKWARD_WITH_UNSHARD(s, 0)`` also unshards parameters and
      returns unsharded parameters and gradients that become gradient
      accumulators.
    - ``FORWARD_BACKWARD_WITH_REDUCE_GRAD(s, N - 1)`` takes unsharded
      parameters and gradient accumulators, accumulates unsharded gradients
      into them in place, and reduces the accumulated gradients.
    - ``FORWARD(s, m)`` runs a PP>1 stage-forward graph.
    - ``BACKWARD_WITH_REDUCE_GRAD(s, m)`` runs a PP>1 stage-backward graph
      containing FSDP gradient reduction.
    - ``BACKWARD(s, m)`` runs a PP>1 stage-backward graph with gradient
      reduction extracted.
    - ``UNSHARD(s)``, ``REDUCE_GRAD(s)``, and ``RESHARD(s)`` are explicit,
      stage-local schedule actions.

    For PP>1, PyTorch pipeline schedules emit ``FULL_BACKWARD``. GraphRuntime
    replaces it with an explicit backward variant before graph construction;
    ``FULL_BACKWARD`` is not part of GraphRuntime IR.

    Runtime lifecycle
    -----------------
    Every path follows the same lifecycle:

    1. Trace or load a joint graph for each local stage.
    2. Apply graph passes.
    3. Keep PP=1 computation joint. Only PP>1 calls ``partition_joint_graph``.
    4. Optionally extract FSDP UNSHARD and REDUCE_GRAD into separate schedule
       actions.
    5. Bind the graphs to schedule actions.
    6. Execute the schedule.
    7. We do not rely on the autograd engine to achieve gradient accumulation.
       PP=1 either accumulates inside the joint graphs or in the runtime. For
       PP>1, gradient accumulation happens in the runtime.
    8. Clear transient stage state.

    ``param.grad`` remains live until the optimizer step. Parameters, buffers,
    saved activations, and transient gradient references in ``stage.state``
    live for one runtime invocation.

    Action state transitions
    ------------------------
    Before the first action, the runtime binds the stage graphs and records the
    live parameters, buffers, and trainable parameters in ``stage.state``.

    - ``FORWARD_BACKWARD_NOGRADACCUM`` stores its returned gradients in
      ``stage.state.unsharded_param_grads``. These tensors become the
      accumulator inputs to later joint graphs.
    - ``FORWARD_BACKWARD`` appends the loss to ``stage.output_chunks`` and
      ``schedule._internal_losses`` and increments the stage backward counter.
      With runtime-owned accumulation, reduced gradients go directly to
      ``param.grad`` while raw gradients accumulate for a later
      ``REDUCE_GRAD``. With graph-owned accumulation, the graph updates the
      first microbatch's gradient tensors in place.
    - ``FORWARD_BACKWARD_WITH_UNSHARD`` initializes the gradient accumulators
      and stores its additional unsharded parameter outputs in
      ``stage.state.unsharded_param_values``.
    - ``FORWARD_BACKWARD_WITH_REDUCE_GRAD`` accumulates its raw gradients,
      reduces the complete accumulators, and returns only optimizer-visible
      sharded gradients in ``stage.state.sharded_param_grads``.
    - ``FORWARD`` waits for and consumes any remote input receive, materializes
      parameter inputs if needed, saves its output and backward values in
      ``stage.fwd_cache[m]``, records last-stage losses, and forwards local
      outputs to the next stage.
    - ``BACKWARD`` and ``BACKWARD_WITH_REDUCE_GRAD`` wait for and consume any
      remote gradient receive, increment the stage backward counter, pop
      ``stage.fwd_cache[m]``, and write input gradients to
      ``stage.bwd_cache[m]`` and, when applicable, the previous local stage.
      The former produces raw, unsharded gradients for a later
      ``REDUCE_GRAD``. The runtime adds them to
      ``stage.state.unsharded_param_grads``. The latter accumulates reduced
      gradients directly into ``param.grad``.
    - ``BACKWARD_INPUT`` performs the input-gradient part of that transition
      and saves weight-backward inputs in
      ``stage.saved_values_for_backward_weight_cache[m]``.
      ``BACKWARD_WEIGHT`` pops that entry and defers gradient reduction;
      ``BACKWARD_WEIGHT_WITH_REDUCE_GRAD`` accumulates directly into
      ``param.grad``.
    - ``OVERLAP_F_B`` applies the same forward and backward transitions to its
      two stages in one multiplexed graph call.
    - ``UNSHARD`` populates ``stage.state.unsharded_param_values``;
      ``RESHARD`` clears it. ``REDUCE_GRAD`` populates
      ``stage.state.sharded_param_grads`` and applies schedule gradient scaling
      once.

    Gradient accumulation ownership
    -------------------------------
    Runtime-owned accumulation keeps each graph result separate from its
    destination.

    Calling convention:
    Runtime-owned accumulation with deferred reduction

    FORWARD_BACKWARD(m) or backward action -> returned_param_grads
    runtime accumulation           -> stage.state.unsharded_param_grads += grads
    REDUCE_GRAD                    -> stage.state.sharded_param_grads
    successful schedule exit       -> param.grad += final_param_grads

    If reduction remains in the compute graph, its reduced result is
    accumulated directly into ``param.grad`` instead of passing through
    ``REDUCE_GRAD``. An empty ``param.grad`` is initialized with a clone because
    the graph may reuse its output storage.

    PP=1 graph-owned accumulation uses the first microbatch gradients as its
    destination buffers. The runtime allocates no accumulator tensors; it only
    carries graph outputs to the next graph inputs.

    Calling convention:
    Graph-owned accumulation

    FORWARD_BACKWARD_NOGRADACCUM(0) -> first_grads
    stage.state.unsharded_param_grads <- first_grads
    FORWARD_BACKWARD(1 ... N - 1) -> first_grads.add_(microbatch_grads)
    optional REDUCE_GRAD -> stage.state.sharded_param_grads
    successful schedule exit -> param.grad += final_param_grads

    WGrad fusion replaces a supported ``producer -> add_`` pair with a producer
    that writes directly into the same accumulator. It changes no runtime state
    transition. Unsupported producers retain the explicit ``add_``.

    Calling convention:
    Graph-owned accumulation with FSDP boundaries in edge microbatches

    FORWARD_BACKWARD_WITH_UNSHARD(0)
        -> loss, first_grads, retained_unsharded_params
        -> stage.state.unsharded_param_grads = first_grads
    FORWARD_BACKWARD(1 ... N - 2)
        -> WGrad/add_ updates first_grads in place
    FORWARD_BACKWARD_WITH_REDUCE_GRAD(N - 1)
        -> WGrad/add_ updates first_grads, then reduce-scatter
        -> stage.state.sharded_param_grads

    On exit, including after an exception, the runtime clears ``stage.state``,
    its bound-graph lookup, and per-call loss arguments. PP>1 evaluation
    performs the same setup and cleanup without committing gradients. PP=1
    ``FORWARD_BACKWARD`` is currently training-only.

    SPMD schedule
    -------------
    With no extracted FSDP boundary and no graph-owned accumulation, PP=1
    retains the original workflow: trace the joint graph, optimize the joint
    graph, and run it once per microbatch as ``FORWARD_BACKWARD``.

    With both FSDP boundaries extracted and unsharded parameters reused across
    microbatches, the schedule is::

        UNSHARD(stage=0)
        FORWARD_BACKWARD_NOGRADACCUM(stage=0, microbatch=0)
        ...
        FORWARD_BACKWARD(stage=0, microbatch=N - 1)
        REDUCE_GRAD(stage=0)
        RESHARD(stage=0)

    If the FSDP reshard policy does not reuse unsharded parameters, each joint
    action is surrounded by ``UNSHARD`` and ``RESHARD`` instead. If either
    boundary is not extracted, its collective remains inside each joint graph
    invocation and its explicit action is absent. ``REDUCE_GRAD`` always runs
    once after the last microbatch when reduction is extracted.

    With first/last-microbatch FSDP boundaries, the schedule is::

        FORWARD_BACKWARD_WITH_UNSHARD(stage=0, microbatch=0)
        FORWARD_BACKWARD(stage=0, microbatch=1)
        ...
        FORWARD_BACKWARD_WITH_REDUCE_GRAD(stage=0, microbatch=N - 1)
        RESHARD(stage=0)

    The first graph receives sharded parameters and returns both unsharded
    values and gradients retained in runtime state for this step. Middle and
    last graphs receive those values. Their WGrad producers update the first
    microbatch gradients directly; the last graph then reduces them.

    Pipeline parallelism
    --------------------
    For PP>1, the upstream schedule owns action ordering and communication.
    This is the only path that partitions the joint graph into forward and
    backward graphs. Gradient accumulation remains runtime-owned because an
    optimizer step may invoke the PP schedule more than once. The state
    transitions above apply independently to each physical or virtual stage.
    Current schedules emit one ``REDUCE_GRAD(s)`` after each stage's final
    backward. ``UNSHARD(s)`` and ``RESHARD(s)`` may run more than once per stage;
    none of these actions is global.

    Local stages exchange forward outputs and input gradients through upstream
    stage caches. The upstream schedule creates and waits for remote P2P
    operations.

    Args:
        stages: Local graph stages. PP=1 requires exactly one stage.
        num_microbatches: Trainer accumulation steps for PP=1, or configured
            pipeline microbatches for PP>1.
        parallel_dims: Parallel topology used to select PP=1 or PP>1 behavior.
        parallelism: Parallel configuration used to construct the schedule.
        compile_config: GraphTrainer execution-mode configuration.
        model_config: Model configuration consumed by graph passes.
        loss_fn: Loss function used by the schedule and graph provider.
        trainer_config: Full Trainer configuration supplied for PP=1. Only the
            ``FORWARD_BACKWARD`` path consumes it; PP>1 supplies ``None``.
    """
    if num_microbatches < 1:
        raise ValueError(
            f"GraphRuntime requires at least one microbatch, got {num_microbatches}"
        )

    pp_enabled = parallel_dims.pp_enabled
    if pp_enabled:
        _validate_graph_pp_config(
            compile_config=compile_config,
            parallelism=parallelism,
        )
    else:
        _validate_spmd_graph_runtime_config(compile_config)
        if len(stages) != 1:
            raise ValueError(f"PP=1 requires one local stage, got {len(stages)}")

    fsdp_policy = resolve_graph_runtime_fsdp_policy(
        compile_config,
        num_microbatches=num_microbatches,
        pp_enabled=pp_enabled,
        fsdp_enabled=parallel_dims.fsdp_enabled,
    )
    gradient_accumulation_policy = resolve_graph_runtime_gradient_accumulation_policy(
        compile_config,
        num_microbatches=num_microbatches,
        pp_enabled=pp_enabled,
        fsdp_enabled=parallel_dims.fsdp_enabled,
        split_fsdp_grad_reduction=fsdp_policy.split_fsdp_grad_reduction,
    )

    if pp_enabled:
        return _make_pipeline_parallel_graph_runtime(
            stages,
            num_microbatches=num_microbatches,
            fsdp_policy=fsdp_policy,
            gradient_accumulation_policy=gradient_accumulation_policy,
            compile_config=compile_config,
            model_config=model_config,
            parallelism=parallelism,
            loss_fn=loss_fn,
            parallel_dims=parallel_dims,
        )

    requires_graph_extraction = (
        fsdp_policy.split_fsdp_param_unshard
        or fsdp_policy.split_fsdp_grad_reduction
        or gradient_accumulation_policy.accumulate_in_graph
    )
    if compile_config.precompile_artifact_dir and requires_graph_extraction:
        raise ValueError(
            "PP=1 precompiled artifacts do not support extracted FSDP "
            "boundaries or in-graph gradient accumulation"
        )
    if trainer_config is None:
        raise ValueError("PP=1 FORWARD_BACKWARD requires Trainer config")
    return _make_spmd_graph_runtime(
        stages[0],
        num_microbatches=num_microbatches,
        fsdp_policy=fsdp_policy,
        gradient_accumulation_policy=gradient_accumulation_policy,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        loss_fn=loss_fn,
        trainer_config=trainer_config,
        parallel_dims=parallel_dims,
    )


def make_spmd_graph_runtime(
    model: nn.Module,
    *,
    gradient_accumulation_steps: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    device: torch.device,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    trainer_config: "GraphTrainer.Config",
) -> GraphRuntime:
    """Represent one SPMD model as a single-stage graph runtime."""
    # PipelineStage treats `group=None` as the world group.
    # TODO: Remove this when PipelineStage supports local single-stage execution.
    pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
    assert pp_mesh is not None
    stage = GraphPipelineStage(
        model,
        stage_index=0,
        num_stages=1,
        device=device,
        group=pp_mesh.get_group("pp"),
    )
    return make_graph_runtime(
        [stage],
        num_microbatches=gradient_accumulation_steps,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        trainer_config=trainer_config,
    )


def graph_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    loss_fn: LossFunction,
) -> tuple[GraphRuntime, list[BaseModel], bool, bool]:
    """Build a GraphPP pipeline schedule for GraphTrainer.

    Args:
        model: The full model before PP stage splitting.
        parallel_dims: TorchTitan parallel dimension helper.
        training: Training config used for local batch size.
        parallelism: Parallelism config used for PP schedule and module split.
        compile_config: GraphTrainer compile config.
        ac_config: Activation checkpointing config forwarded to the model.
        dump_folder: Artifact/debug output directory.
        device: Local device for the stage.
        model_config: Model config consumed by stage graph passes.
        loss_fn: Loss function used by upstream PP metadata and GraphPP tracing.

    Returns:
        A tuple of ``(runtime, model_parts, has_first_stage, has_last_stage)``.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    for index, stage_modules in enumerate(module_names_per_stage):
        logger.debug("GraphPP stage %s modules: %s", index, stage_modules)

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_mesh.get_local_rank(),
        pp_mesh.size(),
        parallelism.pipeline_parallel_schedule,
        len(module_names_per_stage),
    )
    model_parts: list[BaseModel] = []
    stages: list[GraphPipelineStage] = []
    for stage_index in pp_rank_to_stage_indices:
        model_part = _split_module(model, module_names_per_stage[stage_index])
        model_part = model_part.parallelize(
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        logger.info(
            "PP rank %s is building GraphPP stage_idx %s with modules %s",
            pp_mesh.get_local_rank(),
            stage_index,
            module_names_per_stage[stage_index],
        )
        model_parts.append(model_part)
        stages.append(
            GraphPipelineStage(
                model_part,
                stage_index=stage_index,
                num_stages=len(module_names_per_stage),
                device=device,
                group=pp_mesh.get_group("pp"),
                get_mesh=get_mesh_cb,
            )
        )

    graph_runtime = make_graph_runtime(
        stages,
        num_microbatches=parallelism.num_pp_microbatches,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        trainer_config=None,
    )

    return (
        graph_runtime,
        model_parts,
        any(stage.is_first for stage in stages),
        any(stage.is_last for stage in stages),
    )
