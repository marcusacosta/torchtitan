# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect

from torchtitan.config import Configurable


def build_model_config_for_conversion(
    model_name: str, model_flavor: str
) -> Configurable.Config:
    """Build the unparallelized model config used for checkpoint conversion."""
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    build_model_config = model_module.build_model_config
    builder_kwargs = {}
    if "enable_sp" in inspect.signature(build_model_config).parameters:
        # Conversion only needs parameter shapes and FQNs, which are identical
        # for the SP and non-SP shared-expert implementations.
        builder_kwargs["enable_sp"] = False
    return build_model_config(model_flavor, **builder_kwargs)
