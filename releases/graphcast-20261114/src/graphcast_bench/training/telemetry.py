"""Read host device samples and keep JAX allocator statistics separately labeled."""
import json
import math
import threading
import time

from graphcast_bench.inference.telemetry import read_host_sample


class TrainingMonitor:
    def __init__(self, context, allocator_stats):
        self.context = context
        self.allocator_stats = allocator_stats
        self.samples = []
        self.stopped = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def sample(self):
        context = self.context
        key = (
            context.device.get("uuid")
            if context.platform == "h100"
            else context.device["index"]
        )
        value, evidence = read_host_sample(
            context.output, key, platform=context.platform
        )
        memory = None
        if evidence["status"] == "available":
            data = next(iter(json.loads(value).values()))
            memory = float(
                data["memory_mib"]
                if context.platform == "h100"
                else data["vram Total Used Memory (MiB)"]
            )
        allocator = None
        try:
            stats = self.allocator_stats() or {}
            if "bytes_in_use" in stats:
                allocator = float(stats["bytes_in_use"]) / 2**20
                if not math.isfinite(allocator) or allocator < 0:
                    allocator = None
        except Exception:
            pass  # Unavailable allocator counters are not zero usage.
        row = dict(
            time_ns=time.time_ns(),
            device_used_mib=memory,
            allocator_used_mib=allocator,
            device_sample=evidence,
            device_source="host",
            allocator_source="jax_allocator",
        )
        with self.lock:
            self.samples.append(row)
        return row

    def start(self):
        def observe():
            with (self.context.output / "device-memory.jsonl").open("w") as stream:
                while not self.stopped.is_set():
                    try:
                        row = self.sample()
                    except Exception as error:
                        row = dict(time_ns=time.time_ns(), error=repr(error))
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    self.stopped.wait(1)

        self.thread = threading.Thread(
            target=observe, name="training-telemetry", daemon=True
        )
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(5)

    def summary(self):
        with self.lock:
            rows = list(self.samples)

        def peak(key):
            return max((row[key] for row in rows if row[key] is not None), default=None)

        return dict(
            observed_device_peak_mib=peak("device_used_mib"),
            observed_allocator_peak_mib=peak("allocator_used_mib"),
            memory_scope="Device peak: sampled host total; allocator peak: JAX bytes_in_use. Missing counters are null.",
        )
