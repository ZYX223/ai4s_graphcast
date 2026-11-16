import hashlib
import json
from pathlib import Path
import sys


def read_record(path):
    """Reject malformed JSON shapes as invalid evidence, not Python errors."""
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("Checkpoint evidence must be a JSON object: " + str(path))
    if not isinstance(report.get("checkpoint_reload"), dict):
        raise ValueError("Checkpoint reload evidence must be a JSON object: " + str(path))
    for name in ("mode", "data_sha256", "plan_sha256", "checkpoint_sha256"):
        if name in report and (not isinstance(report[name], str) or not report[name]):
            raise ValueError("Invalid checkpoint evidence field: " + name)
    for name in ("tiny", "preflight"):
        if name in report and type(report[name]) is not bool:
            raise ValueError("Invalid checkpoint evidence field: " + name)
    return report


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_updates(report):
    if (not isinstance(report.get("rows"), list) or not report["rows"]
            or not isinstance(report["rows"][-1], dict)):
        raise ValueError("Invalid predecessor update rows")
    update = report["rows"][-1]["update"]
    if type(update) is not int or not 1 <= update <= 38:
        raise ValueError("Invalid predecessor update count")
    rollout = 1 if update <= 16 else 2 + (update - 17) // 2
    if type(report.get("rollout")) is not int or report["rollout"] != rollout:
        raise ValueError("Predecessor rollout does not match its update count")
    return update


def validate_resume(path):
    """Read committed evidence without unpickling; accept historical stage saves."""
    path = Path(path)
    folder = path.parent
    commit = path.with_suffix(".json")
    if commit.exists():
        report = read_record(commit)
        required = {"mode", "tiny", "preflight", "data_sha256", "plan_sha256", "checkpoint_sha256"}
        if (not required.issubset(report)
                or report.get("schema") != "graphcast.training_checkpoint.v1"
                or report.get("status") != "completed"
                or report.get("checkpoint_file") != path.name
                or report.get("mode") not in ("fp32", "tf32", "rne_xf32")
                or report.get("checkpoint_reload", {}).get("status") != "passed"):
            raise ValueError("Checkpoint commit record is invalid: " + str(commit))
        update = completed_updates(report)
        reload = report["checkpoint_reload"]
        if (any(not isinstance(reload.get(key), str) or not reload[key]
                for key in ("params_sha256", "optimizer_sha256"))
                or reload.get("optimizer_counters") != [update]
                or reload.get("device_reload") is not True):
            raise ValueError("Checkpoint reload evidence is incomplete: " + str(commit))
        if file_sha256(path) != report.get("checkpoint_sha256"):
            raise ValueError("Checkpoint bytes differ from committed state")
        return report
    if path.name != "checkpoint.pkl":
        raise ValueError("Checkpoint has no complete commit record: " + str(path))
    report = read_record(folder / "report.json")
    if (
        report.get("status") != "completed"
        or report.get("checkpoint_reload", {}).get("status") != "passed"
    ):
        raise ValueError("Predecessor did not complete checkpoint validation")
    # Historical workers and current delivery adapters use these status files.
    exits = [folder / n for n in ("exit-code.txt", "python-exit-code.txt", "engine-exit-code.txt")
             if (folder / n).exists()]
    if not exits or any(int(p.read_text()) != 0 for p in exits):
        raise ValueError("Predecessor process has no successful exit")
    if file_sha256(path) != report.get("checkpoint_sha256"):
        raise ValueError("Checkpoint bytes differ from recorded successful state")
    return report


def resolve_checkpoint(path):
    """Select the greatest valid update in one run, never in sibling experiments."""
    root = Path(path).resolve()
    if not root.exists():
        raise ValueError("--resume checkpoint does not exist: " + str(root))
    if root.is_file():
        validate_resume(root)
        return root
    candidates = []
    for horizon in range(1, 13):
        stage = root / ("r%02d" % horizon)
        candidates.extend(stage.glob("checkpoints/checkpoint-*.pkl"))
        if (stage / "checkpoint.pkl").exists():
            candidates.append(stage / "checkpoint.pkl")
    valid = []
    for candidate in sorted(candidates):
        try:
            candidate.resolve().relative_to(root)
            # Metadata must belong to this run too, even when a path is a symlink.
            for evidence in (candidate.with_suffix(".json"), candidate.parent / "report.json"):
                if evidence.exists():
                    evidence.resolve().relative_to(root)
            report = validate_resume(candidate)
            valid.append((completed_updates(report), candidate, report))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            print("Skipping invalid checkpoint {}: {}".format(candidate, error), file=sys.stderr)
    if not valid:
        raise ValueError("No valid checkpoints in run directory; specify one run, not --work: " + str(root))
    for key in ("mode", "tiny", "preflight", "data_sha256", "plan_sha256"):
        values = {item[2][key] for item in valid if key in item[2]}
        if len(values) > 1:
            raise ValueError("Mixed checkpoint configurations in run directory: " + key)
    hashes_by_update = {}
    for update, _, report in valid:
        hashes_by_update.setdefault(update, set()).add(report["checkpoint_sha256"])
    for update, hashes in hashes_by_update.items():
        if len(hashes) != 1:
            raise ValueError("Conflicting checkpoints at update %s" % update)
    latest = max(hashes_by_update)
    choices = [item for item in valid if item[0] == latest]
    return choices[0][1]
