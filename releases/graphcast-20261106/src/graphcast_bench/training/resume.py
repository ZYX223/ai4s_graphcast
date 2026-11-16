import hashlib
import json
from pathlib import Path


def validate_resume(path):
    path = Path(path)
    folder = path.parent
    report = json.loads((folder / "report.json").read_text())
    if (
        report.get("status") != "completed"
        or report.get("checkpoint_reload", {}).get("status") != "passed"
    ):
        raise ValueError("Predecessor did not complete checkpoint validation")
    # H100 launcher writes exit-code; Hygon also writes python-exit-code.
    exits = [folder / n for n in ["exit-code.txt", "python-exit-code.txt"] if (folder / n).exists()]
    if not exits or any(int(p.read_text()) != 0 for p in exits):
        raise ValueError("Predecessor process has no successful exit")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("checkpoint_sha256"):
        raise ValueError("Checkpoint bytes differ from recorded successful state")
    return report
