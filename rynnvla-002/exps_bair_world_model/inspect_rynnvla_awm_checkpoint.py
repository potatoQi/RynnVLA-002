#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from safetensors import safe_open


ACTION_HEAD_KEYS = [
    "action_head.action_token_embeddings.weight",
    "action_head.output_projection.model.layer_norm1.weight",
    "action_head.output_projection.model.layer_norm1.bias",
    "action_head.output_projection.model.fc1.weight",
    "action_head.output_projection.model.fc1.bias",
    "action_head.output_projection.model.fc2.weight",
    "action_head.output_projection.model.fc2.bias",
]


def resolve_checkpoint(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    return (Path(__file__).resolve().parent / raw).resolve()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_shape(checkpoint_dir: Path, weight_map: dict, key: str) -> list[int] | None:
    shard_name = weight_map.get(key)
    if shard_name is None:
        return None
    with safe_open(checkpoint_dir / shard_name, framework="pt", device="cpu") as handle:
        return list(handle.get_tensor(key).shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="../ckpts/Action_World_model_512/libero_spatial",
        help="Path to a RynnVLA Action_World_model_512 checkpoint directory.",
    )
    parser.add_argument("--target-action-dim", type=int, default=4)
    parser.add_argument("--target-time-horizon", type=int, default=1)
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint(args.checkpoint)
    index = load_json(checkpoint_dir / "model.safetensors.index.json")
    config = load_json(checkpoint_dir / "config.json")
    train_args = load_json(checkpoint_dir / "args.json")
    weight_map = index.get("weight_map", {})

    hidden_size = int(config.get("hidden_size", 4096))
    expected_action_embedding_width = args.target_action_dim * args.target_time_horizon * hidden_size

    action_head_shapes = {
        key: tensor_shape(checkpoint_dir, weight_map, key)
        for key in ACTION_HEAD_KEYS
    }

    shard_files = sorted({name for name in weight_map.values() if name.endswith(".safetensors")})
    missing_shards = [
        name for name in shard_files
        if not (checkpoint_dir / name).exists()
    ]
    aria2_files = sorted(path.name for path in checkpoint_dir.glob("*.aria2"))

    report = {
        "checkpoint": str(checkpoint_dir),
        "exists": checkpoint_dir.exists(),
        "total_size_bytes_from_index": index.get("metadata", {}).get("total_size"),
        "num_tensors": len(weight_map),
        "shard_files": shard_files,
        "missing_shards": missing_shards,
        "aria2_files": aria2_files,
        "checkpoint_action_dim": train_args.get("action_dim"),
        "checkpoint_time_horizon": train_args.get("time_horizon"),
        "target_action_dim": args.target_action_dim,
        "target_time_horizon": args.target_time_horizon,
        "hidden_size": hidden_size,
        "expected_target_action_embedding_shape": [1, expected_action_embedding_width],
        "checkpoint_action_head_shapes": action_head_shapes,
        "requires_ignore_mismatched_sizes": (
            train_args.get("action_dim") != args.target_action_dim
            or train_args.get("time_horizon") != args.target_time_horizon
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
