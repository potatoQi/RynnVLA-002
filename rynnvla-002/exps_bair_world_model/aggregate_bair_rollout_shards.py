from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sharded BAIR rollout eval metrics.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--keep-shards", action="store_true")
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


def save_contact_sheet(image_paths: list[Path], path: Path, *, max_cols: int = 2) -> bool:
    if not image_paths:
        return False
    try:
        from PIL import Image
    except Exception:
        shutil.copy2(image_paths[0], path)
        return True

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    try:
        cols = max(1, min(max_cols, len(images)))
        rows = math.ceil(len(images) / cols)
        cell_w = max(image.width for image in images)
        cell_h = max(image.height for image in images)
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        for index, image in enumerate(images):
            row, col = divmod(index, cols)
            x = col * cell_w + (cell_w - image.width) // 2
            y = row * cell_h + (cell_h - image.height) // 2
            sheet.paste(image, (x, y))
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return True
    finally:
        for image in images:
            image.close()


def save_horizon_plot(metrics: dict[str, Any], path: Path) -> bool:
    per_horizon = metrics.get("per_horizon", {})
    if not isinstance(per_horizon, dict) or not per_horizon:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    horizons = sorted((int(horizon), values) for horizon, values in per_horizon.items())
    x = [horizon for horizon, _ in horizons]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for axis, key, title in zip(
        axes,
        ("frame_mse", "frame_psnr", "frame_ssim"),
        ("MSE", "PSNR", "SSIM"),
        strict=True,
    ):
        y = [float(values.get(key, 0.0)) for _, values in horizons]
        axis.plot(x, y, marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("horizon")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def merge_artifacts(
    out_dir: Path,
    shard_dirs: list[Path],
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    keep_shards: bool,
) -> dict[str, object]:
    rollout_dir = out_dir / "rollout_videos"
    shutil.rmtree(rollout_dir, ignore_errors=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    shard_metric_paths = []
    shard_manifest_paths = []
    merged_manifests = []
    gif_paths = []
    grid_sources = []

    gif_index = 0
    for shard_index, shard_dir in enumerate(shard_dirs):
        for name in ("metrics.json", "sample_manifest.json"):
            path = shard_dir / name
            if path.exists():
                if name == "metrics.json":
                    shard_metric_paths.append(str(path))
                else:
                    shard_manifest_paths.append(str(path))
                    manifest = load_json(path)
                    if isinstance(manifest, list):
                        merged_manifests.extend(manifest)
        gif_dir = shard_dir / "rollout_videos"
        if gif_dir.exists():
            for gif_path in sorted(gif_dir.glob("*.gif")):
                merged_path = rollout_dir / f"rollout_clip_{gif_index:03d}.gif"
                shutil.copy2(gif_path, merged_path)
                gif_paths.append(str(merged_path))
                gif_index += 1
        grid = shard_dir / "prediction_grid.png"
        if grid.exists():
            grid_sources.append(grid)

    prediction_grid = out_dir / "prediction_grid.png"
    if not save_contact_sheet(grid_sources, prediction_grid):
        prediction_grid.unlink(missing_ok=True)

    horizon_metrics = out_dir / "horizon_metrics.png"
    if not save_horizon_plot(metrics, horizon_metrics):
        horizon_metrics.unlink(missing_ok=True)

    if merged_manifests:
        save_json(out_dir / "sample_manifest.json", merged_manifests)
    save_json(out_dir / "run_config.json", config)

    artifacts: dict[str, object] = {
        "prediction_grid": str(prediction_grid) if prediction_grid.exists() else "",
        "horizon_metrics": str(horizon_metrics) if horizon_metrics.exists() else "",
        "rollout_gifs": gif_paths,
    }
    if keep_shards:
        artifacts["shard_metrics"] = shard_metric_paths
        artifacts["shard_manifests"] = shard_manifest_paths
    else:
        for shard_dir in shard_dirs:
            shutil.rmtree(shard_dir, ignore_errors=True)
    return artifacts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
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
    save_json(
        out_dir / "rollout_artifacts.json",
        merge_artifacts(out_dir, shard_dirs, metrics, config, keep_shards=args.keep_shards),
    )
    print(f"[aggregate] wrote {out_dir}")


if __name__ == "__main__":
    main()
