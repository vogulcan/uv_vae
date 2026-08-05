"""Guard on ``decode_workers`` under the cuDF decode path.

``decode_workers > 1`` is a CPU-path optimisation: it overlaps per-reader decode so the
single-threaded numpy encode+split stop stalling the loader. Its cost there is host RAM.
Under ``UV_VAE_GPU_DECODE=1`` the same concurrency lands in the bounded RMM pool -- 0.25
of a 16 GB budget, so 4 GB by default -- and N row groups in flight multiply into it.
The resulting OOM appears partway through an epoch, hours into a run, which is why this
is enforced in code rather than left to the docstring that used to be the only warning.

These tests need no GPU: the guard keys off the environment variable, not off whether
cuDF imports.
"""

from __future__ import annotations

import pytest

from uv_vae.multi_streaming import (
    ALLOW_GPU_DECODE_WORKERS_ENV,
    _resolve_decode_workers,
)

GPU_DECODE_ENV = "UV_VAE_GPU_DECODE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both switches start unset, so a stray value in the caller's shell cannot leak in."""
    monkeypatch.delenv(GPU_DECODE_ENV, raising=False)
    monkeypatch.delenv(ALLOW_GPU_DECODE_WORKERS_ENV, raising=False)


@pytest.mark.parametrize("requested", [1, 2, 8, 48])
def test_cpu_path_is_untouched(requested: int) -> None:
    """With GPU decode off the setting must pass through: this is the measured default."""
    assert _resolve_decode_workers(requested) == requested


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE", "Yes"])
def test_gpu_decode_clamps_concurrency(truthy: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any spelling that enables GPU decode also clamps, matching gpu_decode_requested."""
    monkeypatch.setenv(GPU_DECODE_ENV, truthy)
    assert _resolve_decode_workers(8) == 1


@pytest.mark.parametrize("falsy", ["0", "", "no", "off"])
def test_falsy_gpu_decode_does_not_clamp(falsy: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GPU_DECODE_ENV, falsy)
    assert _resolve_decode_workers(8) == 8


def test_sequential_request_is_already_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GPU_DECODE_ENV, "1")
    assert _resolve_decode_workers(1) == 1


def test_explicit_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who has measured the pool can keep the concurrency."""
    monkeypatch.setenv(GPU_DECODE_ENV, "1")
    monkeypatch.setenv(ALLOW_GPU_DECODE_WORKERS_ENV, "1")
    assert _resolve_decode_workers(8) == 8


def test_override_alone_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override must not perturb the CPU path it has no business touching."""
    monkeypatch.setenv(ALLOW_GPU_DECODE_WORKERS_ENV, "1")
    assert _resolve_decode_workers(8) == 8


def test_clamp_is_reported(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Silently halving throughput would be worse than the OOM it prevents."""
    monkeypatch.setenv(GPU_DECODE_ENV, "1")
    _resolve_decode_workers(8)
    message = capsys.readouterr().err
    assert "clamping to 1" in message
    assert ALLOW_GPU_DECODE_WORKERS_ENV in message
