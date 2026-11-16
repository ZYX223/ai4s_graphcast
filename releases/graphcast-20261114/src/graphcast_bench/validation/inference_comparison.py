"""CPU-only single-step or 3x40 comparison to the frozen customer H100.

Uses the unchanged project metrics; never reads the old gpu41 output as reference.
Execution completion and numerical acceptance are deliberately checked separately.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

from graphcast_bench.assets import ROOT, verify_files
from graphcast_bench.metrics import compare, final_frame
from graphcast_bench.platforms.native import sources

REFERENCE_ID = "h100-earth-tf32-20261103-v1"
REFERENCE_SHA = "36d97a25fc9f823f5a80ff32e54be294440b663a2decc8d0160661f420ef1cc1"
ROUTER_SHA = "27a3966aba19349480bdd30c2f820a230c12494f1a4634b7c83311b352834f5e"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def check_sha(path, expected):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    require(h.hexdigest() == expected, "Checksum mismatch: " + str(path))
    return h.hexdigest()


def compare_frames(candidate, reference, case):
    result = compare(candidate, reference, case=case)
    result["reference"] = REFERENCE_ID
    result["internal_under_5_percent"] = result.pop("metrics_within_limits")
    result["stage_at_most_10_percent"] = (
        result["overall_mean_percent"] <= 10 and result["z500_percent"] <= 10
    )
    return result


def validate_reports(candidate, reference, steps):
    for name, r, mode, repeats in [
        ("candidate", candidate, "rne_xf32", 1 if steps == 1 else 3),
        ("reference", reference, "tf32", 2 if steps == 1 else 3),
    ]:
        require(r["status"] == "completed", name + " incomplete")
        require(
            (r["measured_steps"], r["repeats"], r["warmup_steps"]) == (steps, repeats, 5),
            name + " wrong steps/repeats/warmup",
        )
        require(
            r["mode"] == mode
            and r["storage"] == "float32"
            and r["jax_matmul_precision"] == "tensorfloat32",
            name + " wrong precision",
        )
        require(len(r["rows"]) == repeats, name + " missing rounds")
        for i, row in enumerate(r["rows"], 1):
            require(row["round"] == i and row["mode"] == mode, name + " wrong round identity")
            require(
                len(row["apply_seconds"]) == steps
                and all(math.isfinite(t) and t > 0 for t in row["apply_seconds"]),
                name + " invalid measured times",
            )
            require(
                row["jit_cache_before"] == row["jit_cache_after"], name + " compiled during timing"
            )
    require(reference["router_build"] is None, "H100 must have no explicit RNE router")
    require(
        candidate["xla_mode"] == "fl_only" and candidate["variant"] == "fused", "Wrong BW backend"
    )
    build = candidate["router_build"]
    require(
        build["schema"] == "graphcast.native.v1" and build["kind"] == "inference",
        "Wrong RNE library kind",
    )
    require(
        set(build["managed_sources"]) == set(sources("inference")),
        "Incomplete native build sources",
    )
    verify_files(ROOT, build["managed_sources"])
    for key in ("source", "data", "original_case_sha256"):
        require(candidate[key] == reference[key], "Different source/data/case: " + key)


def load_frame(path, case):
    raw = xr.load_dataset(path, decode_cf=False)
    require(
        raw.time.attrs.get("units") == "hours"
        and np.array_equal(raw.time.values, [case["lead_hours"]]),
        "Wrong forecast lead: " + str(path),
    )
    raw.time.attrs.pop("dtype", None)
    ds = xr.decode_cf(raw, decode_timedelta=True)
    final_frame(ds, case["initialization_time"], case["lead_hours"])
    return ds


def reference_files(root, steps):
    manifest = root / "reference" / REFERENCE_ID / "REFERENCE.json"
    check_sha(manifest, REFERENCE_SHA)
    frozen = read(manifest)
    require(frozen["reference_id"] == REFERENCE_ID, "Wrong reference ID")
    if steps == 40:
        paths = frozen["inference"]["prediction_files"]
        for item in paths + frozen["inference"]["reports"] + [frozen["inference"]["timing"]]:
            check_sha(root / item["path"], item["sha256"])
        report = read(root / "runs/inference-full40-r1/run/report.json")
        for index, item in enumerate(paths):
            row = report["rows"][index]
            require(
                row["output_sha256"] == item["sha256"],
                "Reference report differs from frozen output",
            )
        return report, [(i, root / item["path"], item["sha256"]) for i, item in enumerate(paths, 1)]
    supplemental = read(root / "single-reference.json")
    require(
        supplemental["parent_reference"] == REFERENCE_ID and supplemental["selected_round"] == 2,
        "Wrong supplemental single-step reference",
    )
    for item in supplemental["files"]:
        check_sha(root / item["path"], item["sha256"])
    report = read(root / "runs/inference-single-r1/run/report.json")
    check = read(root / "runs/inference-single-r1/single-step-validation.json")
    require(check["status"] == "passed", "H100 single-step optimization failed")
    path = root / "runs/inference-single-r1/run/round2-tf32/prediction.nc"
    digest = report["rows"][1]["output_sha256"]
    check_sha(path, digest)
    return report, [(2, path, digest)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate", type=Path, help="Completed BW run directory containing report.json"
    )
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Local mirror root, containing reference/ and runs/",
    )
    p.add_argument("--case", type=Path, default=ROOT / "configs/case.json")
    p.add_argument("--steps", type=int, choices=(1, 40), required=True)
    p.add_argument("--accept-stage-threshold", action="store_true",
                   help="Use the current <=10% stage gate; retain the <5% internal verdict")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--self-check", action="store_true", help="H100 output self-comparison; not BW acceptance"
    )
    a = p.parse_args(argv)
    require(a.self_check != bool(a.candidate), "Select either candidate or self-check")
    require(not a.output.exists(), "Output must be new")
    case = read(a.case)
    require(case["lead_hours"] == 240, "Use the unchanged original 40-step case")
    reference_report, ref_files = reference_files(a.reference, a.steps)
    # Prepared JAX case and upstream source-case have different recorded hashes.
    check_sha(a.case, reference_report["data"]["case_config_sha256"])
    require(
        case["source_case_sha256"] == reference_report["original_case_sha256"],
        "Prepared case derives from a different original case",
    )
    case = dict(case, lead_hours=a.steps * 6)
    if a.self_check:
        candidates = ref_files
    else:
        # Do not accept a partial/cancelled run merely because its exit code is zero.
        from graphcast_bench.validation.completion import validate

        validate(a.candidate.parent, "single" if a.steps == 1 else "full40")
        candidate_report = read(a.candidate / "report.json")
        validate_reports(candidate_report, reference_report, a.steps)
        candidates = [
            (r["round"], a.candidate / r["run"] / "prediction.nc", r["output_sha256"])
            for r in candidate_report["rows"]
        ]
    refs = [(i, load_frame(path, case)) for i, path, digest in ref_files]
    rows, details, inputs = [], [], []
    for ci, path, digest in candidates:
        check_sha(path, digest)
        candidate = load_frame(path, case)
        inputs.append(dict(round=ci, path=str(path), sha256=digest))
        for ri, reference in refs:
            result = compare_frames(candidate, reference, case)
            details.extend(
                dict(candidate_round=ci, reference_round=ri, **row) for row in result.pop("layers")
            )
            rows.append(
                dict(
                    candidate_round=ci,
                    reference_round=ri,
                    primary_pair=(a.steps == 1 or ci == ri),
                    **result,
                )
            )
        del candidate
    summary = dict(
        status="reference_self_check" if a.self_check else "compared",
        reference=REFERENCE_ID,
        measured_steps=a.steps,
        lead_hours=case["lead_hours"],
        inputs=inputs,
        reference_inputs=[dict(round=i, path=str(p), sha256=h) for i, p, h in ref_files],
        rows=rows,
        all_pairs_under_5_percent=all(x["internal_under_5_percent"] for x in rows),
        all_pairs_at_most_10_percent=all(x["stage_at_most_10_percent"] for x in rows),
        max_overall_percent=max(x["overall_mean_percent"] for x in rows),
        max_z500_percent=max(x["z500_percent"] for x in rows),
        metric="Per-field relative MAE, H100 denominator; 6 variables x 13 levels equal weight; Z500 separate.",
        scope="Single +6h screening"
        if a.steps == 1
        else "Independent 40-step rollouts; final +240h only, not weather truth.",
    )
    a.output.mkdir(parents=True)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, data in [("pairs.csv", rows), ("variable-level-details.csv", details)]:
        with (a.output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("rows", "inputs", "reference_inputs")},
            indent=2,
        )
    )
    if a.accept_stage_threshold and not summary["all_pairs_at_most_10_percent"]:
        raise SystemExit("Stage 10% consistency check failed; raw metrics saved")
    if not a.accept_stage_threshold and not summary["all_pairs_under_5_percent"]:
        raise SystemExit(
            "Internal 5% consistency check failed; raw metrics and stage verdict saved"
        )


if __name__ == "__main__":
    main()
