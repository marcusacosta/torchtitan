# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch

from .configs import CompileConfig


_LOCAL_COMPILE_CALLBACKS: dict[
    str, list[Callable[[CompileConfig | None, bool], bool]]
] = {}
logger = logging.getLogger(__name__)


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    options: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently."""

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def apply_local_compile(
            compile_config: CompileConfig | None,
            batch_invariant_mode: bool,
        ) -> bool:
            nonlocal fn
            enabled = (
                compile_config is not None and name in compile_config.local_compile
            )
            # Compile only when enabled and compatible with batch-invariant mode.
            if enabled and (not batch_invariant_mode or batch_invariant):
                fn = torch.compile(
                    reference,
                    backend=compile_config.backend,
                    fullgraph=True,
                    options=options,
                )
            else:
                fn = reference
            return enabled and batch_invariant_mode and not batch_invariant

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(apply_local_compile)
        return wrapped

    return decorate


def configure_local_compile_functions(
    compile_config: CompileConfig | None,
    batch_invariant: bool,
) -> None:
    """Bind registered functions to eager or torch.compile implementations."""
    if compile_config is not None:
        unknown = [
            name
            for name in compile_config.local_compile
            if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown compile.local_compile entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

    batch_invariant_fallbacks: set[str] = set()
    for name, callbacks in _LOCAL_COMPILE_CALLBACKS.items():
        for apply_local_compile_fn in callbacks:
            if apply_local_compile_fn(compile_config, batch_invariant):
                batch_invariant_fallbacks.add(name)

    if batch_invariant_fallbacks:
        logger.warning(
            "Local compile targets %s do not support batch-invariant mode and "
            "will run eagerly.",
            sorted(batch_invariant_fallbacks),
        )


__all__ = ["configure_local_compile_functions", "local_compile"]
