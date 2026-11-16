"""Own exactly one GPU input batch, preserving both gradient evaluations.

The caller gives ownership via a one-element list. It must not keep aliases to
that device batch. Host input trees remain NumPy-backed and unchanged.
"""

import gc
import time


def two_microbatch_grads(
    params,
    device_batches,
    second_host_batch,
    grad,
    transfer,
    synchronize,
    select,
    memory_stats,
    clock=time.perf_counter,
):
    if len(device_batches) != 1:
        raise ValueError("Streaming requires exactly one owned initial device batch")
    batch = device_batches.pop()
    begin = clock()
    select("microbatch-0")
    l0, g0 = grad(params, batch)
    synchronize((l0, g0))
    first_end = clock()
    before_release = memory_stats()
    # Do not assign transfer(...) to the old batch variable: its RHS allocates
    # while the old LHS still exists. Synchronize above, then remove the last
    # owned input reference before starting the next upload.
    del batch
    gc.collect()
    after_release = memory_stats()
    upload_begin = clock()
    batch = transfer(second_host_batch)
    synchronize(batch)
    upload_end = clock()
    after_second_upload = memory_stats()
    second_begin = clock()
    select("microbatch-1")
    l1, g1 = grad(params, batch)
    synchronize((l1, g1))
    second_end = clock()
    device_batches.append(batch)
    return (
        l0,
        g0,
        l1,
        g1,
        dict(
            before_release=before_release,
            after_release=after_release,
            after_second_upload=after_second_upload,
            forward_backward_microbatch_seconds=[first_end - begin, second_end - second_begin],
            second_microbatch_h2d_seconds=upload_end - upload_begin,
            release_and_diagnostics_seconds=(upload_begin - first_end)
            + (second_begin - upload_end),
            scope="Same params for both gradients; first input synchronized and released before second H2D; gradients retained for original averaging and one optimizer update",
        ),
    )
