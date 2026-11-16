"""Corresponding-update BW RNE+XF32 versus frozen new H100 TF32 loss."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

REFERENCE_ID = "h100-earth-tf32-20261103-v1"
REFERENCE_SHA = "36d97a25fc9f823f5a80ff32e54be294440b663a2decc8d0160661f420ef1cc1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare_reports(b, h):
    require(b["mode"] == "rne_xf32" and h["mode"] == "tf32", "Wrong cross-platform precision")
    require(b["status"] == h["status"] == "completed" and bool(b["rows"]), "Incomplete training")
    fixed = [
        "rollout",
        "tiny",
        "preflight",
        "seed",
        "data_sha256",
        "optimizer",
        "model",
        "task",
        "source_revision",
    ]
    if b["rows"][0]["update"] == 1:
        fixed += ["initial_params_sha256", "initial_optimizer_sha256", "initial_batch_sha256"]
    for k in fixed:
        require(b[k] == h[k], "Changed invariant: " + k)
    refs = {r["update"]: r for r in h["rows"]}
    require(len(refs) == len(h["rows"]), "Duplicate H100 updates")
    results = []
    seen = set()
    for row in b["rows"]:
        u = row["update"]
        require(u not in seen and u in refs, "Duplicate/unmatched update")
        seen.add(u)
        ref = refs[u]
        for k in ["rollout", "lr", "starts", "batch_sha256"]:
            require(row[k] == ref[k], "Changed update invariant: " + k)
        x, y = float(row["loss"]), float(ref["loss"])
        require(math.isfinite(x) and math.isfinite(y) and y != 0, "Invalid loss/reference")
        results.append(
            dict(
                update=u,
                horizon=row["rollout"],
                lr=row["lr"],
                bw_loss=x,
                h100_loss=y,
                absolute_difference=abs(x - y),
                relative_error_percent=100 * abs(x - y) / abs(y),
            )
        )
    errors = [r["relative_error_percent"] for r in results]
    return dict(
        reference=REFERENCE_ID,
        rows=results,
        updates=len(results),
        max_relative_error_percent=max(errors),
        mean_relative_error_percent=statistics.mean(errors),
        last_relative_error_percent=errors[-1],
        internal_under_5_percent=all(x < 5 for x in errors),
        stage_at_most_10_percent=all(x <= 10 for x in errors),
        scope="Corresponding update loss only; independent parameter/optimizer trajectories; short validation is not convergence",
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    require(not a.output.exists(), "Comparison output must be new")
    check = json.loads((a.candidate / "completion-validation.json").read_text())
    require(check["status"] == "execution_complete", "Candidate completion not validated")
    require(
        (a.candidate / "python-exit-code.txt").read_text().strip() == "0",
        "Candidate process failed",
    )
    b = json.loads((a.candidate / "report.json").read_text())
    manifest = a.reference_root / "reference" / REFERENCE_ID / "REFERENCE.json"
    require(
        hashlib.sha256(manifest.read_bytes()).hexdigest() == REFERENCE_SHA,
        "Wrong frozen reference manifest",
    )
    m = json.loads(manifest.read_text())
    require(m["reference_id"] == REFERENCE_ID, "Wrong reference ID")
    h_path = a.reference_root / "training" / ("r%02d" % b["rollout"]) / "report.json"
    expected = next(
        x for x in m["training"]["reports"] if Path(x["path"]).parent.name == h_path.parent.name
    )
    require(
        hashlib.sha256(h_path.read_bytes()).hexdigest() == expected["sha256"],
        "H100 report checksum mismatch",
    )
    result = compare_reports(b, json.loads(h_path.read_text()))
    result.update(
        candidate_report_sha256=hashlib.sha256(
            (a.candidate / "report.json").read_bytes()
        ).hexdigest(),
        reference_report_sha256=expected["sha256"],
    )
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0 if result["internal_under_5_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
