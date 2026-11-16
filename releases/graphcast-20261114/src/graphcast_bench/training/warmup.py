"""Warm gradient/optimizer executables without advancing the training state."""

import math

from graphcast_bench.delivery.options import positive_integer


def run_warmup(gradient, average, update, params, state, batch, iterations,
               copy_tree, synchronize):
    """Each iteration discards optimizer results; donated buffers are copies.

    ``update`` binds the existing zero warmup learning rate. ``gradient`` does
    not donate its inputs. Both callbacks use the same math as formal updates.
    """
    positive_integer("warmup_iters", iterations)
    for _ in range(iterations):
        loss, grads = gradient(params, batch)
        synchronize((loss, grads))
        if not math.isfinite(float(loss)):
            raise ValueError("Warmup loss must be finite")
        averaged = average(copy_tree(grads), grads)
        warm_params, warm_state = copy_tree((params, state))
        result = update(warm_params, warm_state, averaged)
        synchronize(result)
        del loss, grads, averaged, warm_params, warm_state, result
