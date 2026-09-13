#!/usr/bin/env python3
"""Summarize legacy GraphCast probe metrics (not formal training segments)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "loss" not in item or "step" not in item:
                raise ValueError(f"{path}:{line_no} must contain `step` and `loss`")
            rows.append(item)
    if not rows:
        raise ValueError(f"no metrics found in {path}")
    rows.sort(key=lambda item: int(item["step"]))
    return rows


def maybe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in rows if item.get(key) is not None]
    if not values:
        return None
    return mean(values)


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# GraphCast Training Loss Summary",
        "",
        f"- metrics: `{summary['metrics']}`",
        f"- requested first N: `{summary['requested_first_n']}`",
        f"- available steps: `{summary['available_steps']}`",
        f"- summarized steps: `{summary['summarized_steps']}`",
        f"- first step: `{summary['first_step']}`",
        f"- last summarized step: `{summary['last_summarized_step']}`",
        f"- first loss: `{summary['first_loss']}`",
        f"- final summarized loss: `{summary['final_summarized_loss']}`",
        f"- mean loss: `{summary['mean_loss']}`",
        f"- min loss: `{summary['min_loss']}`",
        f"- max loss: `{summary['max_loss']}`",
        f"- mean grad_mean_abs: `{summary['mean_grad_mean_abs']}`",
        f"- mean grad_max_abs: `{summary['mean_grad_max_abs']}`",
        "",
    ]
    if summary["summarized_steps"] < summary["requested_first_n"]:
        lines.extend(
            [
                "Note: this run has fewer rows than the requested first-N window,",
                "so the mean is computed over all available rows.",
                "",
            ]
        )
    return "\n".join(lines)


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--first-n", type=int, default=1000)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    if args.first_n <= 0:
        raise ValueError("--first-n must be positive")

    metrics_path = Path(args.metrics)
    rows = load_metrics(metrics_path)
    selected = rows[: args.first_n]
    losses = [float(item["loss"]) for item in selected]
    summary = {
        "metrics": str(metrics_path),
        "requested_first_n": args.first_n,
        "available_steps": len(rows),
        "summarized_steps": len(selected),
        "first_step": int(selected[0]["step"]),
        "last_summarized_step": int(selected[-1]["step"]),
        "first_loss": losses[0],
        "final_summarized_loss": losses[-1],
        "mean_loss": mean(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "mean_grad_mean_abs": maybe_mean(selected, "grad_mean_abs"),
        "mean_grad_max_abs": maybe_mean(selected, "grad_max_abs"),
    }
    json_path = Path(args.json_output)
    report_path = Path(args.report)
    json_dump(json_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")

    print(f"metrics={metrics_path}")
    print(f"summarized_steps={summary['summarized_steps']}")
    print(f"mean_loss={summary['mean_loss']}")
    print(f"json={json_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
