from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sharded BAIR rollout eval metrics.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sample_count(metrics: dict[str, Any]) -> float:
    return float(metrics.get("config", {}).get("samples", 0.0))


def weighted_average(shard_metrics: list[dict[str, Any]], group: str) -> dict[str, float]:
    weights = [sample_count(metrics) for metrics in shard_metrics]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {}
    keys = sorted(
        {
            key
            for metrics in shard_metrics
            for key, value in metrics.get(group, {}).items()
            if isinstance(value, (int, float))
        }
    )
    out: dict[str, float] = {}
    for key in keys:
        values = [metrics.get(group, {}).get(key) for metrics in shard_metrics]
        present = [(float(value), weight) for value, weight in zip(values, weights) if isinstance(value, (int, float))]
        if not present:
            continue
        weight_sum = sum(weight for _, weight in present)
        out[key] = sum(value * weight for value, weight in present) / max(weight_sum, 1e-12)

    for key in list(out):
        if key.endswith("_psnr"):
            mse_key = key[: -len("_psnr")] + "_mse"
            if mse_key in out:
                out[key] = 10.0 * math.log10(1.0 / max(out[mse_key], 1e-8))
    return out


def aggregate_per_horizon(shard_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    horizons = sorted(
        {
            horizon
            for metrics in shard_metrics
            for horizon in metrics.get("per_horizon", {}).keys()
        },
        key=lambda item: int(item),
    )
    out: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        horizon_metrics = []
        for metrics in shard_metrics:
            copied = dict(metrics)
            copied["quality"] = metrics.get("per_horizon", {}).get(horizon, {})
            horizon_metrics.append(copied)
        out[horizon] = weighted_average(horizon_metrics, "quality")
    return out


def collect_artifacts(shard_dirs: list[Path]) -> dict[str, object]:
    metric_paths = []
    manifest_paths = []
    gif_paths = []
    grid_paths = []
    horizon_paths = []
    for shard_dir in shard_dirs:
        for name in ("metrics.json", "sample_manifest.json"):
            path = shard_dir / name
            if path.exists():
                if name == "metrics.json":
                    metric_paths.append(str(path))
                else:
                    manifest_paths.append(str(path))
        gif_dir = shard_dir / "rollout_videos"
        if gif_dir.exists():
            gif_paths.extend(str(path) for path in sorted(gif_dir.glob("*.gif")))
        grid = shard_dir / "prediction_grid.png"
        if grid.exists():
            grid_paths.append(str(grid))
        horizon = shard_dir / "horizon_metrics.png"
        if horizon.exists():
            horizon_paths.append(str(horizon))
    return {
        "shard_metrics": metric_paths,
        "shard_manifests": manifest_paths,
        "rollout_gifs": gif_paths,
        "prediction_grids": grid_paths,
        "horizon_metrics": horizon_paths,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    shard_dirs = [Path(path) for path in args.shard_dirs]
    shard_metrics = [load_json(shard_dir / "metrics.json") for shard_dir in shard_dirs]
    first = shard_metrics[0]
    total_samples = sum(sample_count(metrics) for metrics in shard_metrics)
    failures = {
        "decode_failures": sum(float(metrics.get("failures", {}).get("decode_failures", 0.0)) for metrics in shard_metrics),
        "clips_with_failures": sum(float(metrics.get("failures", {}).get("clips_with_failures", 0.0)) for metrics in shard_metrics),
    }
    future_frames = float(first.get("config", {}).get("future_frames", 0.0))
    failures["decode_failure_rate"] = failures["decode_failures"] / max(total_samples * future_frames, 1.0)

    config = dict(first.get("config", {}))
    config["samples"] = total_samples
    config["shard_count"] = float(len(shard_metrics))
    config.pop("shard_index", None)

    metrics = {
        "run": {
            **first.get("run", {}),
            "status": "aggregated",
            "shards": float(len(shard_metrics)),
        },
        "config": config,
        "quality": weighted_average(shard_metrics, "quality"),
        "per_horizon": aggregate_per_horizon(shard_metrics),
        "fvd": {
            "note": "FVD is not aggregated across shards here; run single-process FVD on saved videos if needed."
        },
        "failures": failures,
        "diagnostics": first.get("diagnostics", {}),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "rollout_artifacts.json", collect_artifacts(shard_dirs))
    print(f"[aggregate] wrote {out_dir}")


if __name__ == "__main__":
    main()
