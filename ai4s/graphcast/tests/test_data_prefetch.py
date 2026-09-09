from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import threading
import time
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
package = types.ModuleType("graphcast_compat")
package.__path__ = [str(SOURCE_ROOT / "graphcast_compat")]
sys.modules["graphcast_compat"] = package
spec = importlib.util.spec_from_file_location(
    "graphcast_compat.data_prefetch",
    SOURCE_ROOT / "graphcast_compat/data_prefetch.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

DepthOneCpuPrefetcher = module.DepthOneCpuPrefetcher
OfficialMicrobatchDataRequest = module.OfficialMicrobatchDataRequest
OfficialUpdateCpuProducer = module.OfficialUpdateCpuProducer
OfficialUpdateDataKey = module.OfficialUpdateDataKey
OfficialUpdateDataRequest = module.OfficialUpdateDataRequest


class DepthOneCpuPrefetcherTests(unittest.TestCase):
    def test_preserves_key_and_rejects_a_second_outstanding_request(self):
        prefetcher = DepthOneCpuPrefetcher(lambda value: value * 2)
        self.addCleanup(prefetcher.close)

        prefetcher.submit((1, 0, 0), 7)
        with self.assertRaisesRegex(RuntimeError, "already has"):
            prefetcher.submit((2, 0, 1), 8)
        with self.assertRaisesRegex(RuntimeError, "training cursor"):
            prefetcher.take((9, 9, 9))

        self.assertEqual(prefetcher.take((1, 0, 0)), 14)
        prefetcher.submit((2, 0, 1), 8)
        self.assertEqual(prefetcher.take((2, 0, 1)), 16)

    def test_worker_exception_returns_to_caller_and_worker_closes(self):
        def fail(_: object) -> object:
            raise ValueError("data preparation failed")

        prefetcher = DepthOneCpuPrefetcher(fail)
        prefetcher.submit("update-1", object())
        with self.assertRaisesRegex(ValueError, "data preparation failed"):
            prefetcher.take("update-1")
        prefetcher.close()

        self.assertFalse(prefetcher.worker_alive)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            prefetcher.submit("update-2", object())

    def test_close_drains_one_running_request_without_leaving_worker(self):
        entered = threading.Event()
        release = threading.Event()

        def bounded_worker(value: int) -> int:
            entered.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test worker was not released")
            return value

        prefetcher = DepthOneCpuPrefetcher(bounded_worker)
        prefetcher.submit("update-1", 1)
        self.assertTrue(entered.wait(timeout=1.0))
        release.set()
        prefetcher.close()

        self.assertFalse(prefetcher.worker_alive)
        self.assertFalse(prefetcher.has_pending)

    def test_single_producer_never_runs_requests_concurrently(self):
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def worker(value: int) -> int:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return value

        with DepthOneCpuPrefetcher(worker) as prefetcher:
            for value in range(4):
                prefetcher.submit(value, value)
                self.assertEqual(prefetcher.take(value), value)

        self.assertEqual(maximum_active, 1)


class OfficialUpdateCpuProducerTests(unittest.TestCase):
    def test_keeps_request_order_and_prepares_only_on_cpu(self):
        calls = []

        class Dataset:
            def sample_indices(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    inputs="inputs",
                    targets="targets",
                    forcings="forcings",
                    rollout_steps=kwargs["target_rollout_steps"],
                    sample_weights=tuple(kwargs["sample_weights"]),
                    global_sample_weights=tuple(kwargs["sample_weights"]),
                    window_indices=tuple(kwargs["window_indices"]),
                    global_window_indices=tuple(kwargs["window_indices"]),
                    rank=kwargs["rank"],
                    valid_sample_count=1,
                    padded_sample_count=1,
                    global_valid_sample_count=1,
                    global_padded_sample_count=1,
                    reference_datetimes=("2017-01-01T00:00:00",),
                    target_datetimes_by_sample=(
                        (
                            "2017-01-01T06:00:00",
                            "2017-01-01T12:00:00",
                            "2017-01-01T18:00:00",
                        ),
                    ),
                    dataset_kind="era5",
                    training_stage="era5_pretrain",
                    dataset_rollout_steps=12,
                )

        observed_devices = []

        def prepare(*args, **kwargs):
            del args
            observed_devices.append(
                (kwargs["input_device"], kwargs["storage_device"])
            )
            return SimpleNamespace(marker=len(observed_devices))

        request = OfficialUpdateDataRequest(
            key=OfficialUpdateDataKey(
                global_step=8,
                epoch=0,
                batch_index=7,
                rollout_steps=3,
                rank=1,
                replicas=2,
                allow_replacement=True,
                microbatches=(
                    OfficialMicrobatchDataRequest(
                        index=0,
                        window_indices=(5, 1),
                        sample_weights=(1.0, 1.0),
                    ),
                    OfficialMicrobatchDataRequest(
                        index=1,
                        window_indices=(9, 2),
                        sample_weights=(1.0, 0.0),
                    ),
                ),
            ),
        )
        producer = OfficialUpdateCpuProducer(
            dataset=Dataset(),
            config=object(),
            stats=object(),
            per_variable_weights={"temperature": 1.0},
            prepare_batch=prepare,
        )
        result = producer(request)

        self.assertEqual(result.key, request.key)
        self.assertEqual(result.rollout_steps, 3)
        self.assertEqual(
            tuple(part.index for part in result.microbatches),
            (0, 1),
        )
        self.assertTrue(
            all(
                not hasattr(part, "sample")
                for part in result.microbatches
            )
        )
        self.assertEqual(
            result.microbatches[0].sample_metadata.provenance[
                "global_window_indices"
            ],
            [5, 1],
        )
        self.assertEqual(
            tuple(call["window_indices"] for call in calls),
            ((5, 1), (9, 2)),
        )
        self.assertTrue(all(call["rank"] == 1 for call in calls))
        self.assertTrue(all(call["replicas"] == 2 for call in calls))
        self.assertTrue(
            all(
                input_device == "cpu"
                and storage_device == "cpu"
                for input_device, storage_device in observed_devices
            )
        )


if __name__ == "__main__":
    unittest.main()
