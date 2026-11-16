"""One independent H100 forecast; compilation is separate from predictor timing."""
import time
from typing import NamedTuple

from graphcast_bench.progress import emit, phase


class RolloutResult(NamedTuple):
    step_seconds: list
    wall_seconds: float
    final_prediction: object
    first_inputs: dict


class RolloutRunner:
    def __init__(self, apply, inputs, targets, forcings, synchronize, audit):
        self.apply, self.inputs = apply, inputs
        self.targets, self.forcings = targets, forcings
        self.synchronize, self.audit = synchronize, audit
        self.compiled = False
        self.compile_seconds = 0.0

    def run(self, label, steps, round_id, repeats, warmup=False):
        import jax
        from graphcast import rollout

        times, first = [], None

        def timed_apply(**kwargs):
            nonlocal first
            if first is None:
                first = kwargs
            if not self.compiled:
                if not warmup:
                    raise RuntimeError('Predictor must compile during warmup')
                begin = time.perf_counter()
                with phase('Compile', 'Compile H100 predictor'):
                    self.apply.lower(**kwargs).compile()
                self.compile_seconds += time.perf_counter() - begin
                self.compiled = True
            begin = time.perf_counter()
            result = self.apply(**kwargs)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - begin
            times.append(elapsed)
            self.audit.append(dict(label=label, step=len(times), seconds=elapsed))
            emit('Warmup' if warmup else 'Inference',
                 'round={}/{} | step={}/{} | predictor={:.3f}s'.format(
                     round_id, repeats, len(times), steps, elapsed))
            return result

        if self.synchronize() != 0:
            raise RuntimeError('CUDA synchronization failed')
        begin, last = time.perf_counter(), None
        # Reset both original inputs and RNG for every warmup/formal invocation.
        for chunk in rollout.chunked_prediction_generator(
                predictor_fn=timed_apply, rng=jax.random.PRNGKey(0), inputs=self.inputs,
                targets_template=self.targets.isel(time=slice(0, steps)),
                forcings=self.forcings.isel(time=slice(0, steps)), num_steps_per_chunk=1):
            last = chunk
        jax.block_until_ready(last)
        wall = time.perf_counter() - begin
        if last is None or len(times) != steps:
            raise RuntimeError('Incomplete H100 rollout')
        return RolloutResult(times, wall, last, first)
