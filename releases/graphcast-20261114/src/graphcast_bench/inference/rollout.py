"""One independent rollout, with shared JIT compilation and explicit step timing."""
import time
from typing import NamedTuple

import jax
import xarray as xr
from graphcast import rollout
from graphcast_bench.progress import emit, phase


class RolloutResult(NamedTuple):
    step_seconds: list
    wall_seconds: float
    final_prediction: xr.Dataset
    first_inputs: dict


class RolloutRunner:
    """Reuse the compiled predictor, never a preceding rollout's input state."""

    def __init__(self, apply, inputs, targets, forcings, router, synchronize):
        self.apply = apply
        self.inputs, self.targets, self.forcings = inputs, targets, forcings
        self.router, self.synchronize = router, synchronize
        self.step_audit = []
        self._compiled = False

    def ensure_compiled(self, sample):
        """Compile from the first real warmup sample before enabling strict routing."""
        if self._compiled:
            return
        with phase('Compile', 'Compile the predictor (excluded from measured prediction time)'):
            self.apply.lower(**sample).compile()
            self.router.library.graphcast_runtime_enable()
            self._compiled = True

    def run_rollout(self, mode, label, steps, *, round_id=1, repeats=1, warmup=False):
        self.router.select(mode, label)
        step_seconds, first_inputs = [], None

        def timed_apply(**inputs):
            nonlocal first_inputs
            if first_inputs is None:
                first_inputs = inputs
            self.ensure_compiled(inputs)
            if warmup and not step_seconds:
                emit('Warmup', 'Starting round={}/{}: steps={}, excluded from measured prediction time'.format(
                    round_id, repeats, steps))
            before = self.router.snapshot()
            started = time.perf_counter()
            prediction = self.apply(**inputs)
            jax.block_until_ready(prediction)
            elapsed = time.perf_counter() - started
            step_seconds.append(elapsed)
            emit('Warmup' if warmup else 'Inference',
                 'round={}/{} | step={}/{} | predictor={:.3f}s'.format(
                     round_id, repeats, len(step_seconds), steps, elapsed))
            # Native audit time remains outside the synchronized predictor timer.
            counts = self.router.verify(before, mode)
            self.step_audit.append(dict(label=label, step=len(step_seconds), seconds=elapsed, **counts))
            return prediction

        assert self.synchronize() == 0
        started = time.perf_counter()
        final_prediction = None
        # The official generator copies input state. A fresh generator and RNG on
        # every call reset both warmup-to-formal and round-to-round trajectories.
        for prediction in rollout.chunked_prediction_generator(
            predictor_fn=timed_apply,
            rng=jax.random.PRNGKey(0),
            inputs=self.inputs,
            targets_template=self.targets.isel(time=slice(0, steps)),
            forcings=self.forcings.isel(time=slice(0, steps)),
            num_steps_per_chunk=1,
        ):
            final_prediction = prediction
        jax.block_until_ready(final_prediction)
        wall_seconds = time.perf_counter() - started
        assert final_prediction is not None and len(step_seconds) == steps
        return RolloutResult(step_seconds, wall_seconds, final_prediction, first_inputs)
