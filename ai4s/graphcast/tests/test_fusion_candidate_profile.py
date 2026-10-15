from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = GRAPHCAST_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphcast_compat.fusion_candidate_profile import (  # noqa: E402
    PostLayerNormResidualRecorder,
)
from graphcast_compat.google_small import (  # noqa: E402
    GoogleSmallConfig,
    GoogleSmallGraphNet,
)


def test_disabled_post_layernorm_recorder_ignores_observations() -> None:
    recorder = PostLayerNormResidualRecorder(output_path=None)
    x = torch.empty((2, 3, 4), dtype=torch.float32)

    with recorder.region(x, torch.empty_like(x), (4,), torch.empty(4), torch.empty(4), 1e-5):
        pass

    assert recorder.groups == ()
    assert recorder.stop() is None


def test_post_layernorm_recorder_groups_without_item_or_synchronize(monkeypatch) -> None:
    recorder = PostLayerNormResidualRecorder(output_path="events.json")
    x = torch.empty((2, 3, 4), dtype=torch.float32)
    residual = torch.empty_like(x)
    monkeypatch.setattr(torch.Tensor, "item", lambda _: pytest.fail("item must not be called"))

    recorder.observe(x, residual, (4,), torch.empty(4), torch.empty(4), 1e-5)

    assert recorder.groups[0].count == 1
    assert recorder.groups[0].shape == (2, 3, 4)
    assert recorder.groups[0].x_stride == (12, 4, 1)
    assert recorder.groups[0].residual_stride == (12, 4, 1)


def test_post_layernorm_recorder_rejects_mixed_devices() -> None:
    recorder = PostLayerNormResidualRecorder(output_path="events.json")

    with pytest.raises(ValueError, match="same device"):
        recorder.observe(
            torch.empty(4),
            torch.empty(4, device="meta"),
            (4,),
            torch.empty(4),
            torch.empty(4),
            1e-5,
        )


class _FakeEvent:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp
        self.recorded = False
        self.streams: list[object | None] = []

    def record(self, stream: object | None = None) -> None:
        self.recorded = True
        self.streams.append(stream)

    def elapsed_time(self, end: _FakeEvent) -> float:
        assert self.recorded and end.recorded
        return end.timestamp - self.timestamp


def test_post_layernorm_recorder_defers_accelerator_event_resolution(tmp_path: Path) -> None:
    events = iter((_FakeEvent(1.0), _FakeEvent(4.5)))
    synchronized: list[torch.device] = []
    output = tmp_path / "events.json"
    recorder = PostLayerNormResidualRecorder(
        output,
        event_pair_factory=lambda device: (next(events), next(events)),
        synchronize_device=synchronized.append,
    )
    x = torch.empty((2, 3, 4))

    with recorder.region(x, torch.empty_like(x), (4,), torch.empty(4), torch.empty(4), 1e-5):
        assert synchronized == []

    assert synchronized == []
    payload = recorder.stop(metadata={"run": "fake"})

    assert synchronized == [torch.device("cpu")]
    assert payload["groups"] == [
        {
            "count": 1,
            "device": "cpu",
            "device_time_total_ms": 3.5,
            "dtype": "torch.float32",
            "eps": 1e-5,
            "normalized_shape": [4],
            "residual_stride": [12, 4, 1],
            "shape": [2, 3, 4],
            "timed_count": 1,
            "weight_shape": [4],
            "x_stride": [12, 4, 1],
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"] == {"run": "fake"}


class _FakeTensor:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.shape = torch.Size((2, 3, 4))
        self.dtype = torch.float32

    def stride(self) -> tuple[int, ...]:
        return (12, 4, 1)


def test_post_layernorm_recorder_records_on_the_candidate_device_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (_FakeEvent(1.0), _FakeEvent(4.5))
    requested_devices: list[torch.device] = []
    stream = object()
    current_device = torch.device("cuda:0")
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: (requested_devices.append(device), stream)[1],
    )
    recorder = PostLayerNormResidualRecorder(
        tmp_path / "events.json",
        event_pair_factory=lambda device: events,
    )
    x = _FakeTensor("cuda:1")
    residual = _FakeTensor("cuda:1")

    with recorder.region(x, residual, (4,), _FakeTensor("cuda:1"), None, 1e-5):
        pass

    assert x.device != current_device
    assert requested_devices == [torch.device("cuda:1")]
    assert events[0].streams == [stream]
    assert events[1].streams == [stream]


def test_candidate_observer_wraps_only_node_update_post_layernorm_residual() -> None:
    observed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def observer(x, residual, normalized_shape, weight, bias, eps):
        observed.append((tuple(x.shape), tuple(residual.shape)))
        return PostLayerNormResidualRecorder.disabled_region()

    config = GoogleSmallConfig(
        latent_size=4,
        hidden_layers=1,
        gnn_msg_steps=1,
        node_input_dim=4,
        edge_input_dim=1,
        output_dim=1,
    )
    graph_net = GoogleSmallGraphNet(
        edge_types=("edge",),
        edge_node_sets={"edge": ("node", "node")},
        node_types=("node",),
        node_input_dims={"node": 4},
        edge_input_dims={"edge": 1},
        node_processor_input_dims={"node": 8},
        edge_processor_input_dims={"edge": 12},
        node_output_dims=None,
        extra_final_node_output_dims=None,
        edge_update_chunk_sizes=None,
        node_update_chunk_sizes=None,
        decoder_node_chunk_sizes=None,
        message_passing_steps=1,
        config=config,
        post_layer_norm_residual_observer=observer,
    )

    graph_net(
        {"node": torch.randn(1, 3, 4)},
        {"edge": torch.randn(1, 4, 1)},
        {"edge": (torch.tensor([0, 1, 2, 0]), torch.tensor([1, 2, 0, 2]))},
    )

    assert observed == [((1, 3, 4), (1, 3, 4))]
