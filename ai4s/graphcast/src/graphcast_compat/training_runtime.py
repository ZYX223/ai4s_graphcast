"""Legacy diagnostic state hashing only; formal execution is the epoch runner.

Retained until the source-segment diagnostic migration. No training loop lives here.
"""
from .training_state import state_object_sha256


def training_start_record(session, metadata):
    """One small CPU-readable comparison record per invocation, not per update.

    Reuse the existing resume contract; hash only parameters and optimizer state,
    not the large static graph buffers. This runs outside update timing.
    """
    return {"event": "start", **metadata, "loss_comparison": {
        "start_optimizer_step": session.cursor.global_optimizer_step,
        "starting_parameters_sha256": state_object_sha256(dict(session.model.named_parameters())),
        "starting_optimizer_sha256": state_object_sha256(session.optimizer.state_dict()),
        "resume_contract": session.resume_contract,
    }}
