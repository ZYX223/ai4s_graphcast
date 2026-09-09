from __future__ import annotations

import importlib.util
from enum import Enum
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "graphcast_compat"
TEST_PACKAGE = "graphcast_profile_testpkg"
if TEST_PACKAGE not in sys.modules:
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[TEST_PACKAGE] = package


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profiled_mfu = load_module(
    f"{TEST_PACKAGE}.profiled_mfu",
    PACKAGE_ROOT / "profiled_mfu.py",
)
ascend_profile = load_module(
    f"{TEST_PACKAGE}.ascend_profile",
    PACKAGE_ROOT / "ascend_profile.py",
)
hotspot_profiler = load_module(
    f"{TEST_PACKAGE}.hotspot_profiler",
    PACKAGE_ROOT / "hotspot_profiler.py",
)


class ProfiledMfuTests(unittest.TestCase):
    def write_trace(self, events: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return path

    def test_extracts_each_profiled_step_and_unions_kernels_per_device(self):
        trace_path = self.write_trace(
            [
                {
                    "name": "ProfilerStep#0",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 0,
                    "dur": 20,
                },
                {
                    "name": "ProfilerStep#0",
                    "cat": "gpu_user_annotation",
                    "ph": "X",
                    "ts": 1,
                    "dur": 5,
                },
                {
                    "name": "kernel-a",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 0,
                    "dur": 10,
                    "args": {"device": 0},
                },
                {
                    "name": "kernel-b-overlaps-a",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 5,
                    "dur": 10,
                    "args": {"device": 0},
                },
                {
                    "name": "kernel-c",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 2,
                    "dur": 3,
                    "args": {"device": 1},
                },
                {
                    "name": "copy-is-not-compute",
                    "cat": "gpu_memcpy",
                    "ph": "X",
                    "ts": 5,
                    "dur": 12,
                    "args": {"device": 1},
                },
                {
                    "name": "ProfilerStep#2",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 40,
                    "dur": 20,
                },
                {
                    "name": "kernel-d",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 42,
                    "dur": 4,
                    "args": {"device": 0},
                },
                {
                    "name": "kernel-e",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 48,
                    "dur": 8,
                    "args": {"device": 1},
                },
            ]
        )

        result = profiled_mfu.extract_profiled_step_busy(
            trace_path, expected_devices=("0", "1")
        )

        first = result[1]
        self.assertEqual(first["device_busy_us_by_device"], {"0": 15.0, "1": 3.0})
        self.assertEqual(first["total_device_busy_us"], 18.0)
        self.assertEqual(first["device_busy_ms"], 0.009)
        self.assertEqual(first["profile_step_wall_ms"], 0.02)
        self.assertLessEqual(first["device_busy_ms"], first["profile_step_wall_ms"])

        third = result[3]
        self.assertEqual(third["device_busy_us_by_device"], {"0": 4.0, "1": 8.0})
        self.assertEqual(third["device_busy_ms"], 0.006)

    def test_repairs_torch_musa_empty_process_group_description_in_memory(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "musa-trace.json"
        malformed = (
            '{"traceEvents":['
            '{"name":"ProfilerStep#0","cat":"user_annotation","ph":"X",'
            '"ts":0,"dur":20},'
            '{"name":"kernel","cat":"kernel","ph":"X","ts":2,"dur":3,'
            '"args":{"device":0,"Process Group Description": ,"unused":1}}'
            "]}"
        )
        path.write_text(malformed, encoding="utf-8")

        result = profiled_mfu.extract_profiled_step_busy(
            path, expected_devices=("0",)
        )

        self.assertEqual(result[1]["total_device_busy_us"], 3.0)
        self.assertIn(
            '"Process Group Description": ,',
            path.read_text(encoding="utf-8"),
        )

    def test_stdlib_repairs_tokens_across_read_boundaries(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "split-token.json"
        malformed = (
            '{"traceEvents":[{"name":"first"},'
            '{"name":"kernel","args":{"Process Group Description": ,'
            '"unicode":"气象"}},{"name":"last"}]}'
        )
        path.write_text(malformed, encoding="utf-8")
        expected = [
            {"name": "first"},
            {"name": "kernel", "args": {
                "Process Group Description": None, "unicode": "气象"}},
            {"name": "last"},
        ]
        for chunk_size in range(1, 96):
            with self.subTest(chunk_size=chunk_size), mock.patch.object(
                profiled_mfu, "_TRACE_READ_CHUNK_BYTES", chunk_size
            ):
                self.assertEqual(
                    list(profiled_mfu._iter_trace_events_stdlib(path)), expected
                )
        self.assertEqual(path.read_text(encoding="utf-8"), malformed)

    def test_streams_torch_npu_top_level_trace_and_unions_ascend_compute(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace_view.json"
        events = [
            {
                "ph": "M",
                "name": "process_name",
                "pid": 104,
                "tid": 0,
                "args": {"name": "Ascend Hardware"},
            },
            {
                "ph": "M",
                "name": "process_name",
                "pid": 105,
                "tid": 0,
                "args": {"name": "Ascend Hardware"},
            },
            {
                "name": "ProfilerStep#7",
                "cat": "cpu_op",
                "ph": "X",
                "ts": "100.0",
                "dur": "40.0",
                "pid": 1,
                "tid": 1,
                "args": {},
            },
            {
                "name": "MatMul",
                "ph": "X",
                "ts": "102.0",
                "dur": "10.0",
                "pid": 104,
                "tid": 3,
                "args": {"Task Type": "AI_CORE", "Device Id": 0},
            },
            {
                "name": "Vector",
                "ph": "X",
                "ts": "106.0",
                "dur": "8.0",
                "pid": 104,
                "tid": 4,
                "args": {"Task Type": "AI_VECTOR_CORE", "Device Id": 0},
            },
            {
                "name": "MatMul",
                "ph": "X",
                "ts": "110.0",
                "dur": "20.0",
                "pid": 105,
                "tid": 3,
                "args": {"Task Type": "MIX_AIC", "Device Id": 1},
            },
            {
                "name": "DMA",
                "ph": "X",
                "ts": "101.0",
                "dur": "30.0",
                "pid": 105,
                "tid": 5,
                "args": {"Task Type": "SDMA_SQE", "Device Id": 1},
            },
        ]
        path.write_text(json.dumps(events), encoding="utf-8")

        result = profiled_mfu.extract_profiled_step_busy(
            path, expected_devices=("0", "1")
        )

        # ProfilerStep#7 is the eighth runtime update. Overlapping AIC/AIV
        # kernels on the first device are unioned; DMA is deliberately absent.
        self.assertEqual(result[8]["device_busy_us_by_device"], {"0": 12.0, "1": 20.0})
        self.assertEqual(result[8]["total_device_busy_us"], 32.0)

    def test_streaming_reader_rejects_truncated_event(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.json.tmp"
        path.write_text(
            '{"traceEvents":[{"name":"ProfilerStep#0","ph":"X"',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "inside a trace event"):
            list(profiled_mfu.iter_trace_events(path))

    def test_enrichment_does_not_reuse_busy_time_for_unprofiled_steps(self):
        metrics = [
            {
                "step": 1,
                "step_seconds": 0.020,
                "end_to_end_step_seconds": 0.025,
                "training_flops": 1.0e12,
                "recompute_flops": 0.5e12,
                "inference_only": False,
            },
            {
                "step": 2,
                "step_seconds": 0.030,
                "end_to_end_step_seconds": 0.050,
                "training_flops": 1.0e12,
                "recompute_flops": 0.25e12,
                "inference_only": False,
            },
            {
                "step": 3,
                "step_seconds": 0.040,
                "end_to_end_step_seconds": 0.080,
                "training_flops": 1.0e12,
                "recompute_flops": 0.0,
                "inference_only": False,
            },
        ]
        profiled_steps = {
            1: {
                "device_busy_us_by_device": {"0": 6_000.0, "1": 4_000.0},
                "total_device_busy_us": 10_000.0,
                "device_busy_ms": 5.0,
                "profile_step_wall_ms": 20.0,
                "profile_step_index": 0,
            },
            3: {
                "device_busy_us_by_device": {"0": 10_000.0, "1": 14_000.0},
                "total_device_busy_us": 24_000.0,
                "device_busy_ms": 12.0,
                "profile_step_wall_ms": 40.0,
                "profile_step_index": 2,
            },
        }

        profiled_mfu.enrich_step_metrics(
            metrics,
            profiled_steps=profiled_steps,
            device_count=2,
            peak_tflops_per_device=100.0,
        )

        self.assertEqual(metrics[0]["device_busy_ms"], 5.0)
        self.assertAlmostEqual(metrics[0]["mfu_busy"], 1.0)
        self.assertAlmostEqual(metrics[0]["mfu_sys"], 0.25)
        self.assertAlmostEqual(metrics[0]["mfu_model_sys"], 0.25)
        self.assertAlmostEqual(metrics[0]["hfu_sys"], 0.375)
        self.assertAlmostEqual(metrics[0]["mfu_e2e"], 0.20)
        self.assertEqual(metrics[0]["device_sys_ms"], 20.0)
        self.assertEqual(metrics[0]["total_device_sys_ms"], 40.0)
        self.assertEqual(metrics[0]["device_e2e_ms"], 25.0)
        self.assertEqual(metrics[0]["total_device_e2e_ms"], 50.0)

        self.assertIsNone(metrics[1]["device_busy_ms"])
        self.assertIsNone(metrics[1]["mfu_busy"])
        self.assertAlmostEqual(metrics[1]["mfu_sys"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics[1]["mfu_model_sys"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics[1]["hfu_sys"], 1.25 / 6.0)
        self.assertAlmostEqual(metrics[1]["mfu_e2e"], 0.10)

        self.assertEqual(metrics[2]["device_busy_ms"], 12.0)
        self.assertAlmostEqual(metrics[2]["mfu_busy"], 1.0 / 2.4)
        self.assertAlmostEqual(metrics[2]["mfu_sys"], 0.125)
        self.assertAlmostEqual(metrics[2]["mfu_model_sys"], 0.125)
        self.assertAlmostEqual(metrics[2]["hfu_sys"], 0.125)
        self.assertAlmostEqual(metrics[2]["mfu_e2e"], 0.0625)
        self.assertNotEqual(
            metrics[0]["device_busy_ms"], metrics[2]["device_busy_ms"]
        )

    def test_zero_peak_preserves_raw_performance_and_marks_mfu_unknown(self):
        metrics = [
            {
                "step": 1,
                "step_seconds": 2.0,
                "end_to_end_step_seconds": 3.0,
                "training_flops": 8.0e12,
                "recompute_flops": 2.0e12,
                "inference_only": False,
            }
        ]
        profiled_steps = {
            1: {
                "device_busy_us_by_device": {
                    "0": 1_000_000.0,
                    "1": 1_500_000.0,
                },
                "total_device_busy_us": 2_500_000.0,
                "device_busy_ms": 1_250.0,
                "profile_step_wall_ms": 2_000.0,
                "profile_step_index": 0,
            }
        }

        profiled_mfu.enrich_step_metrics(
            metrics,
            profiled_steps=profiled_steps,
            device_count=2,
            peak_tflops_per_device=0.0,
        )

        metric = metrics[0]
        self.assertAlmostEqual(metric["achieved_tflops"], 4.0)
        self.assertAlmostEqual(metric["total_device_busy_ms"], 2_500.0)
        self.assertIsNone(metric["peak_tflops_per_device"])
        self.assertIsNone(metric["aggregate_peak_tflops"])
        self.assertIsNone(metric["mfu_busy"])
        self.assertIsNone(metric["mfu_sys"])
        self.assertIsNone(metric["hfu_sys"])
        self.assertIsNone(metric["mfu_e2e"])

    def test_trace_callback_exports_active_cycle_immediately(self):
        class FakeProfiler:
            def export_chrome_trace(self, path: str) -> None:
                Path(path).write_text(
                    json.dumps({"traceEvents": []}),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as directory:
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=Path(directory),
                ),
                hotspot_profiler.torch.device("cpu"),
            )

            profiler._on_trace_ready(FakeProfiler())

            trace_path = profiler.latest_trace_path()
            self.assertEqual(trace_path, Path(directory) / "trace.json")
            self.assertTrue(trace_path.is_file())

    def test_trace_callback_recovers_complete_tmp_via_local_staging(self):
        class TmpOnlyProfiler:
            def export_chrome_trace(self, path: str) -> None:
                Path(f"{path}.tmp").write_text(
                    json.dumps({"traceEvents": []}),
                    encoding="utf-8",
                )
                raise OSError("simulated atomic rename failure")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "ossfs-output"
            staging_dir = Path(directory) / "local-staging"
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=output_dir,
                    staging_dir=staging_dir,
                ),
                hotspot_profiler.torch.device("cpu"),
            )

            profiler._on_trace_ready(TmpOnlyProfiler())

            trace_path = output_dir / "trace.json"
            self.assertEqual(json.loads(trace_path.read_text()), {"traceEvents": []})
            self.assertEqual(list(staging_dir.iterdir()), [])
            metadata = json.loads(
                (output_dir / "profiler_backend.json").read_text()
            )
            self.assertEqual(
                metadata["completed_traces"][0]["bytes"],
                trace_path.stat().st_size,
            )

    def test_split_schedule_profiles_updates_8_through_11_separately(self):
        class Action(Enum):
            NONE = 0
            WARMUP = 1
            RECORD = 2
            RECORD_AND_SAVE = 3

        schedule = hotspot_profiler._split_active_schedule(
            SimpleNamespace(ProfilerAction=Action),
            wait=6,
            warmup=1,
            active=4,
            repeat=1,
        )

        self.assertEqual([schedule(step) for step in range(6)], [Action.NONE] * 6)
        self.assertEqual(schedule(6), Action.WARMUP)
        self.assertEqual(
            [schedule(step) for step in range(7, 11)],
            [Action.RECORD_AND_SAVE] * 4,
        )
        self.assertEqual(schedule(11), Action.NONE)

    def test_npu_device_selects_torch_npu_profiler_and_npu_activity(self):
        class Action(Enum):
            NONE = 0
            WARMUP = 1
            RECORD = 2
            RECORD_AND_SAVE = 3

        class Activity(Enum):
            CPU = 0
            NPU = 1

        fake_runtime_profiler = SimpleNamespace(
            ProfilerAction=Action,
            ProfilerActivity=Activity,
        )
        fake_torch_npu = SimpleNamespace(profiler=fake_runtime_profiler)
        with mock.patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            api, backend = hotspot_profiler._profiler_api_for_device(
                SimpleNamespace(type="npu")
            )
            activities = hotspot_profiler.activities_for_device(
                SimpleNamespace(type="npu"), api
            )

        self.assertIs(api, fake_runtime_profiler)
        self.assertEqual(backend, "torch_npu.profiler")
        self.assertEqual(activities, [Activity.CPU, Activity.NPU])

    def test_npu_level1_profiler_requests_hf32_execution_metadata(self):
        calls = []

        class ExperimentalConfig:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_api = SimpleNamespace(
            _ExperimentalConfig=ExperimentalConfig,
            ExportType=SimpleNamespace(Db="db"),
            ProfilerLevel=SimpleNamespace(Level1="level1"),
            AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
        )
        config = hotspot_profiler.HotspotProfilerConfig(
            npu_profiler_level="Level1",
            npu_op_attr=True,
        )

        result = hotspot_profiler._npu_experimental_config(fake_api, config)

        self.assertIsInstance(result, ExperimentalConfig)
        self.assertEqual(
            calls,
            [
                {
                    "export_type": "db",
                    "profiler_level": "level1",
                    "aic_metrics": "pipe",
                    "op_attr": True,
                    "data_simplification": False,
                }
            ],
        )

    def test_npu_level0_profiler_preserves_existing_export_contract(self):
        calls = []

        class ExperimentalConfig:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_api = SimpleNamespace(
            _ExperimentalConfig=ExperimentalConfig,
            ExportType=SimpleNamespace(Db="db"),
        )

        hotspot_profiler._npu_experimental_config(
            fake_api,
            hotspot_profiler.HotspotProfilerConfig(),
        )

        self.assertEqual(calls, [{"export_type": "db"}])

    def test_npu_hotspots_come_from_ascend_hardware_events(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            trace_path = output_dir / "trace.json"
            trace_path.write_text(
                json.dumps(
                    [
                        {
                            "ph": "M",
                            "name": "process_name",
                            "pid": 4,
                            "tid": 0,
                            "args": {"name": "Ascend Hardware"},
                        },
                        {
                            "ph": "X",
                            "name": "MatMul",
                            "pid": 4,
                            "tid": 1,
                            "ts": "100",
                            "dur": "12.5",
                            "args": {"Task Type": "AI_CORE", "Device Id": 0},
                        },
                        {
                            "ph": "X",
                            "name": "MatMul",
                            "pid": 4,
                            "tid": 1,
                            "ts": "120",
                            "dur": "7.5",
                            "args": {"Task Type": "AI_CORE", "Device Id": 0},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=output_dir,
                ),
                SimpleNamespace(type="npu"),
            )
            profiler._trace_paths = [trace_path]

            profiler._write_npu_trace_outputs(trace_path)

            rows = json.loads((output_dir / "hotspots.json").read_text())
            self.assertEqual(rows[0]["name"], "MatMul")
            self.assertEqual(rows[0]["calls"], 2)
            self.assertEqual(rows[0]["device_time_total_us"], 20.0)
            self.assertEqual(
                rows[0]["profiler_source"],
                "torch_npu Ascend Hardware trace",
            )

    def test_npu_callback_registers_raw_capture_without_traceview_export(self):
        class RawProfiler:
            def __init__(self, raw_path: Path) -> None:
                self.prof_if = SimpleNamespace(prof_path=str(raw_path))

            def export_chrome_trace(self, unused: str) -> None:
                raise AssertionError("NPU callback must not invoke TraceView export")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "hotspot_profile"
            raw_path = output_dir / "raw" / "rank_0_123_ascend_pt"
            (raw_path / "PROF_000001_test").mkdir(parents=True)
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=output_dir,
                    active=4,
                ),
                SimpleNamespace(type="npu"),
            )
            profiler._step_calls = 7

            profiler._on_trace_ready(RawProfiler(raw_path))

            self.assertEqual(profiler.trace_paths(), ())
            self.assertEqual(profiler.raw_profile_paths(), (raw_path.resolve(),))
            manifest = json.loads(
                (output_dir / "npu_raw_profiles.json").read_text()
            )
            self.assertEqual(manifest["captures"][0]["local_step"], 8)
            self.assertFalse(manifest["traceview_export_during_training"])

    def test_ascend_sqlite_busy_and_hotspots_need_no_chrome_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw_capture"
            profile_container = raw_path / "PROF_000001_test"
            profile_container.mkdir(parents=True)
            database = profile_container / "msprof_test.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE TASK (
                    startNs INTEGER, endNs INTEGER, deviceId INTEGER,
                    connectionId INTEGER, globalTaskId INTEGER,
                    globalPid INTEGER, taskType INTEGER, contextId INTEGER,
                    streamId INTEGER, taskId INTEGER, modelId INTEGER
                );
                CREATE INDEX TaskIndex ON TASK (startNs, globalTaskId);
                CREATE TABLE COMPUTE_TASK_INFO (
                    name INTEGER, globalTaskId INTEGER PRIMARY KEY,
                    blockDim INTEGER, mixBlockDim INTEGER, taskType INTEGER,
                    opType INTEGER, inputFormats INTEGER,
                    inputDataTypes INTEGER, inputShapes INTEGER,
                    outputFormats INTEGER, outputDataTypes INTEGER,
                    outputShapes INTEGER, attrInfo INTEGER, opState INTEGER,
                    hf32Eligible INTEGER
                );
                CREATE TABLE STRING_IDS (id INTEGER, value TEXT);
                CREATE TABLE SESSION_TIME_INFO (
                    startTimeNs INTEGER, endTimeNs INTEGER
                );
                INSERT INTO SESSION_TIME_INFO VALUES (0, 100000);
                INSERT INTO STRING_IDS VALUES (1, 'MatMul'), (2, 'Vector');
                INSERT INTO COMPUTE_TASK_INFO(name, globalTaskId)
                    VALUES (1, 10), (2, 11), (1, 12);
                INSERT INTO TASK(startNs, endNs, deviceId, globalTaskId)
                    VALUES
                    (0, 20000, 0, 10),
                    (10000, 30000, 0, 11),
                    (5000, 15000, 1, 12),
                    (0, 90000, 1, 999);
                """
            )
            connection.commit()
            connection.close()
            output_dir = Path(directory) / "outputs"

            profile, rows = ascend_profile.analyze_ascend_profile(
                raw_path,
                expected_devices=("0", "1"),
                runtime_step=8,
                output_dir=output_dir,
                top_k=20,
            )

            self.assertEqual(
                profile["device_busy_us_by_device"],
                {"0": 30.0, "1": 10.0},
            )
            self.assertEqual(profile["total_device_busy_us"], 40.0)
            self.assertEqual(profile["device_busy_ms"], 0.02)
            self.assertEqual(profile["profile_step_wall_ms"], 0.1)
            self.assertEqual(rows[0]["name"], "MatMul")
            self.assertEqual(rows[0]["calls"], 2)
            self.assertTrue((output_dir / "profile_update_00000008.json").is_file())
            self.assertFalse(any(output_dir.glob("trace*.json")))
            aggregated = ascend_profile.postprocess_ascend_profiles(
                [{"path": str(raw_path), "local_step": 8}],
                expected_devices=("0", "1"),
                output_dir=output_dir,
                top_k=20,
            )
            self.assertEqual(sorted(aggregated), [8])
            self.assertTrue((output_dir / "hotspots_aggregate.json").is_file())
            self.assertTrue((output_dir / "hotspots_aggregate.txt").is_file())

    def test_ascend_sqlite_uses_compute_task_type_when_metadata_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw_capture"
            profile_container = raw_path / "PROF_000001_test"
            profile_container.mkdir(parents=True)
            database = profile_container / "msprof_test.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE TASK (
                    startNs INTEGER, endNs INTEGER, deviceId INTEGER,
                    connectionId INTEGER, globalTaskId INTEGER,
                    globalPid INTEGER, taskType INTEGER, contextId INTEGER,
                    streamId INTEGER, taskId INTEGER, modelId INTEGER
                );
                CREATE TABLE COMPUTE_TASK_INFO (
                    name INTEGER, globalTaskId INTEGER PRIMARY KEY,
                    blockDim INTEGER, mixBlockDim INTEGER, taskType INTEGER,
                    opType INTEGER, inputFormats INTEGER,
                    inputDataTypes INTEGER, inputShapes INTEGER,
                    outputFormats INTEGER, outputDataTypes INTEGER,
                    outputShapes INTEGER, attrInfo INTEGER, opState INTEGER,
                    hf32Eligible INTEGER
                );
                CREATE TABLE STRING_IDS (id INTEGER, value TEXT);
                CREATE TABLE SESSION_TIME_INFO (
                    startTimeNs INTEGER, endTimeNs INTEGER
                );
                INSERT INTO SESSION_TIME_INFO VALUES (0, 100000);
                INSERT INTO STRING_IDS VALUES
                    (1, 'MatMul'), (2, 'AI_CORE'),
                    (3, 'SDMA_SQE'), (4, 'YES');
                INSERT INTO COMPUTE_TASK_INFO(
                    name, globalTaskId, hf32Eligible
                ) VALUES (1, 10, 4);
                INSERT INTO TASK(
                    startNs, endNs, deviceId, globalTaskId, taskType
                ) VALUES
                    (0, 20000, 0, 10, 2),
                    (5000, 25000, 1, 20, 2),
                    (0, 90000, 1, 21, 3);
                """
            )
            connection.commit()
            connection.close()

            profile, rows = ascend_profile.analyze_ascend_profile(
                raw_path,
                expected_devices=("0", "1"),
                runtime_step=8,
                output_dir=Path(directory) / "outputs",
                top_k=20,
            )

            self.assertEqual(
                profile["device_busy_us_by_device"],
                {"0": 20.0, "1": 20.0},
            )
            self.assertEqual(
                profile["task_type_fallback_compute_event_count_by_device"],
                {"0": 0, "1": 1},
            )
            self.assertEqual(profile["hf32_eligible_task_counts"], {"YES": 1})
            fallback = next(
                row for row in rows if row["name"] == "[task_type] AI_CORE"
            )
            self.assertEqual(fallback["calls"], 1)
            self.assertEqual(
                fallback["metadata_sources"], ["TASK.taskType fallback"]
            )
            self.assertEqual(
                fallback["hf32_eligible_task_counts"],
                {"METADATA_UNAVAILABLE": 1},
            )

    def test_profiler_stops_as_soon_as_a_scheduled_trace_is_ready(self):
        class FakeProfiler:
            def __init__(self) -> None:
                self.stop_calls = 0

            def stop(self) -> None:
                self.stop_calls += 1

        with tempfile.TemporaryDirectory() as directory:
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=Path(directory),
                ),
                hotspot_profiler.torch.device("cpu"),
            )
            fake = FakeProfiler()
            profiler._profiler = fake
            profiler._started = True
            profiler._trace_paths = [Path(directory) / "trace.json"]
            profiler._write_outputs = lambda unused: None

            stopped = profiler.stop_if_trace_ready()

            self.assertTrue(stopped)
            self.assertEqual(fake.stop_calls, 1)
            self.assertFalse(profiler._started)
            self.assertIsNone(profiler._profiler)
            self.assertFalse(profiler.stop_if_trace_ready())
            self.assertEqual(fake.stop_calls, 1)

    def test_profiler_keeps_running_before_the_scheduled_trace_is_ready(self):
        profiler = hotspot_profiler.HotspotProfiler(
            hotspot_profiler.HotspotProfilerConfig(enabled=True),
            hotspot_profiler.torch.device("cpu"),
        )
        profiler._profiler = object()
        profiler._started = True

        stopped = profiler.stop_if_trace_ready()

        self.assertFalse(stopped)
        self.assertTrue(profiler._started)

    def test_profiler_record_is_noop_after_bounded_capture_stops(self):
        profiler = hotspot_profiler.HotspotProfiler(
            hotspot_profiler.HotspotProfilerConfig(enabled=True),
            hotspot_profiler.torch.device("cpu"),
        )
        with mock.patch.object(
            hotspot_profiler.torch.profiler,
            "record_function",
        ) as record_function:
            with profiler.record("after-stop"):
                pass
        record_function.assert_not_called()

    def test_aggregate_hotspots_combine_all_active_runtime_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = hotspot_profiler.HotspotProfiler(
                hotspot_profiler.HotspotProfilerConfig(
                    enabled=True,
                    output_dir=Path(directory),
                    wait=6,
                    warmup=1,
                    active=4,
                    repeat=1,
                    top_k=30,
                ),
                hotspot_profiler.torch.device("cpu"),
            )
            profiler._hotspot_rows_by_trace = [
                [
                    {
                        "name": "kernel",
                        "calls": 2,
                        "cpu_time_total_us": None,
                        "self_cpu_time_total_us": 1.0,
                        "device_time_total_us": 10.0,
                        "self_device_time_total_us": 8.0,
                        "flops": None,
                    }
                ]
                for _ in range(4)
            ]

            profiler._write_aggregate_hotspots()

            rows = json.loads(
                (Path(directory) / "hotspots_aggregate.json").read_text()
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["calls"], 8)
            self.assertEqual(rows[0]["device_time_total_us"], 40.0)
            self.assertEqual(rows[0]["self_device_time_total_us"], 32.0)
            self.assertEqual(rows[0]["profiled_runtime_steps"], [8, 9, 10, 11])

    def test_split_profiler_waits_for_every_active_trace_before_stop(self):
        class FakeProfiler:
            def __init__(self) -> None:
                self.stop_calls = 0

            def stop(self) -> None:
                self.stop_calls += 1

        profiler = hotspot_profiler.HotspotProfiler(
            hotspot_profiler.HotspotProfilerConfig(
                enabled=True,
                active=4,
                repeat=1,
                split_active_traces=True,
            ),
            hotspot_profiler.torch.device("cpu"),
        )
        fake = FakeProfiler()
        profiler._profiler = fake
        profiler._started = True
        profiler._trace_paths = [Path(f"trace-{index}.json") for index in range(3)]

        self.assertFalse(profiler.stop_if_trace_ready())
        profiler._trace_paths.append(Path("trace-3.json"))
        self.assertTrue(profiler.stop_if_trace_ready())
        self.assertFalse(profiler.stop_if_trace_ready())
        self.assertEqual(fake.stop_calls, 1)

if __name__ == "__main__":
    unittest.main()
