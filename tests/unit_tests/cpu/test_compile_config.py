# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch

from torchtitan.components.loss import cross_entropy_loss, mse_loss
from torchtitan.config import (
    CompileConfig,
    configure_local_compile_functions,
    local_compile,
)


@pytest.fixture(autouse=True)
def reset_local_compile():
    configure_local_compile_functions(None, False)
    yield
    configure_local_compile_functions(None, False)


def test_compile_config_default() -> None:
    config = CompileConfig()
    assert config.local_compile == ["gated_rmsnorm"]


def test_compile_config_loss_only() -> None:
    config = CompileConfig(local_compile=["loss"])
    assert config.local_compile == ["loss"]


def test_compile_config_empty_local_compile() -> None:
    config = CompileConfig(local_compile=[])
    assert config.local_compile == []


def test_configure_local_compile_functions_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        configure_local_compile_functions(CompileConfig(local_compile=["foo"]), False)


def test_local_compile_binds_once_from_existing_config(monkeypatch, caplog) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))

        def compiled(value: int) -> tuple[str, int]:
            return "compiled", value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    configure_local_compile_functions(
        CompileConfig(local_compile=["test_local_compile"]),
        False,
    )

    assert fn(3) == ("compiled", 3)
    assert len(compiled_calls) == 1
    assert compiled_calls[0][1]["backend"] == "inductor"
    assert compiled_calls[0][1]["fullgraph"] is True

    with caplog.at_level(logging.WARNING):
        configure_local_compile_functions(
            CompileConfig(local_compile=["test_local_compile"]),
            True,
        )
    assert fn(3) == ("eager", 3)
    assert "test_local_compile" in caplog.text


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    configure_local_compile_functions(CompileConfig(local_compile=["loss"]), False)

    assert compiled_names == [cross_entropy_loss.__name__, mse_loss.__name__]
