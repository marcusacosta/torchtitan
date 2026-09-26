# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

from .configs import CommConfig, CompileConfig, DebugConfig, TrainingConfig
from .configurable import Configurable
from .function import Function
from .local_compile import configure_local_compile_functions, local_compile
from .manager import ConfigManager
from .override import (
    apply_overrides,
    clear_overrides,
    derive,
    Override,
    override,
    OverrideConfig,
)

__all__ = [
    "ConfigManager",
    "Configurable",
    "Function",
    "configure_local_compile_functions",
    "local_compile",
    "TORCH_DTYPE_MAP",
    # Config dataclasses
    "CompileConfig",
    "CommConfig",
    "TrainingConfig",
    "DebugConfig",
    # Override mechanism
    "OverrideConfig",
    "Override",
    "override",
    "derive",
    "apply_overrides",
    "clear_overrides",
]
