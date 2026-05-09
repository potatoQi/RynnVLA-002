from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOWER_IS_BETTER_HINTS = ("mse", "lpips", "fvd", "failure", "failures")
HIGHER_IS_BETTER_HINTS = ("psnr", "ssim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two grouped BAIR rollout metrics.json files.")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--candidate-name", type=str, default="candidate")
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="Allow comparing smoke/non-formal rollout metrics.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def direction_for_key(key: str) -> str:
    lowered = key.lower()
    if any(hint in lowered for hint in LOWER_IS_BETTER_HINTS):
        return "lower"
    if any(hint in lowered for hint in HIGHER_IS_BETTER_HINTS):
        return "higher"
    return "neutral"


def compare_metrics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_flat: dict[str, float] = {}
    cand_flat: dict[str, float] = {}
    flatten("", baseline, base_flat)
    flatten("", candidate, cand_flat)

    rows: list[dict[str, float | str]] = []
    for key in sorted(set(base_flat) & set(cand_flat)):
        base = base_flat[key]
        cand = cand_flat[key]
        delta = cand - base
        rel = delta / abs(base) if base != 0 else float("nan")
        direction = direction_for_key(key)
        improved = ""
        if direction == "lower":
            improved = "yes" if delta < 0 else "no" if delta > 0 else "tie"
        elif direction == "higher":
            improved = "yes" if delta > 0 else "no" if delta < 0 else "tie"
        rows.append(
            {
                "metric": key,
                "baseline": base,
                "candidate": cand,
                "delta": delta,
                "relative_delta": rel,
                "direction": direction,
                "improved": improved,
            }
        )
    return {"rows": rows}


def validate_formal_metrics(name: str, metrics: dict[str, Any], allow_nonformal: bool) -> None:
    diagnostics = metrics.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return
    formal = diagnostics.get("formal_rollout_candidate", True)
    if formal or allow_nonformal:
        return
    kind = diagnostics.get("checkpoint_kind", "unknown")
    warning = diagnostics.get("warning", "")
    raise ValueError(
        f"{name} metrics are marked non-formal ({kind}); refuse to compare as a formal result. "
        f"{warning} Pass --allow-nonformal for smoke/debug comparison."
    )


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if value != value:
        return "nan"
    return f"{value:.6g}"


def write_markdown(path: Path, comparison: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        f"# BAIR Rollout Comparison",
        "",
        f"baseline: `{args.baseline_name}`",
        f"candidate: `{args.candidate_name}`",
        "",
        "| metric | baseline | candidate | delta | rel delta | better | improved |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    interesting_prefixes = (
        "quality.",
        "per_horizon.",
        "fvd.",
        "failures.",
    )
    for row in comparison["rows"]:
        metric = str(row["metric"])
        if not metric.startswith(interesting_prefixes):
            continue
        lines.append(
            "| {metric} | {baseline} | {candidate} | {delta} | {relative_delta} | {direction} | {improved} |".format(
                metric=metric,
                baseline=fmt(row["baseline"]),
                candidate=fmt(row["candidate"]),
                delta=fmt(row["delta"]),
                relative_delta=fmt(row["relative_delta"]),
                direction=row["direction"],
                improved=row["improved"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    validate_formal_metrics(args.baseline_name, baseline, args.allow_nonformal)
    validate_formal_metrics(args.candidate_name, candidate, args.allow_nonformal)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_metrics(baseline, candidate)
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(out_dir / "comparison.md", comparison, args)
    print(f"wrote {out_dir.resolve()}")


if __name__ == "__main__":
    main()
