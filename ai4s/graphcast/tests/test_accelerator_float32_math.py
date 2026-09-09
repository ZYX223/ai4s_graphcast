from __future__ import annotations

from graphcast_compat import backend
from graphcast_compat import accelerator


def test_configure_float32_math_delegates_resolved_backend(monkeypatch) -> None:
    observed: list[tuple[bool, str]] = []

    monkeypatch.setattr(
        accelerator,
        "resolve_accelerator_type",
        lambda requested: "musa",
    )
    monkeypatch.setattr(
        backend,
        "configure_float32_math",
        lambda enabled, device_type: observed.append((enabled, device_type)) or "tf32",
    )

    assert accelerator.configure_float32_math(True, "requested") == "tf32"
    assert observed == [(True, "musa")]
