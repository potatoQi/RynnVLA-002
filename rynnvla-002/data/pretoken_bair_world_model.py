import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from traceback import format_exc

import numpy as np
from PIL import Image
import torch


rynnvla_dir = os.path.abspath(__file__).rsplit("/", 2)[0]
repo_dir = str(Path(rynnvla_dir).parent)
sys.path.extend([repo_dir, rynnvla_dir])

from data.dataset import BairRobotPushingConversation
from data.item_processor import FlexARItemProcessor_Action, var_center_crop


def _load_done_ids(record_path: Path) -> set[int]:
    done = set()
    if not record_path.exists():
        return done
    with record_path.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in item:
                done.add(int(item["id"]))
    return done


def _aggregate_records(out_dir: Path, save_name: str) -> None:
    records = {}
    for path in sorted(out_dir.glob("*-of-*-record.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in item:
                    continue
                records[int(item["id"])] = item

    merged = [records[key] for key in sorted(records)]
    save_path = out_dir / save_name
    with save_path.open("w") as f:
        json.dump(merged, f)
    print(f"wrote {len(merged)} records to {save_path}")


def _build_dataset(args: argparse.Namespace) -> BairRobotPushingConversation:
    return BairRobotPushingConversation(
        args.config,
        resolution=args.target_size,
        with_transition_tokens=args.with_transition_tokens,
        transition_token_id=args.transition_token_id,
        transition_token_count=args.transition_token_count,
    )


def _copy_media(media: dict) -> dict:
    return {
        "input_ids": list(media["input_ids"]),
        "labels": list(media["labels"]),
    }


@torch.no_grad()
def _process_images_batched(item_processor: FlexARItemProcessor_Action, images: list[Image.Image]) -> list[dict]:
    if not images:
        return []

    cropped = [var_center_crop(image, crop_size_list=item_processor.crop_size_list) for image in images]
    tensors = []
    for image in cropped:
        np_img = np.array(image.convert("RGB")) / 255.0
        np_img = np_img * 2 - 1
        tensors.append(torch.from_numpy(np_img).permute(2, 0, 1))

    weight = item_processor.chameleon_ori_image_tokenizer._vq_model.encoder.conv_in.weight
    batch = torch.stack(tensors, dim=0).to(weight)
    _, _, [_, _, img_toks] = item_processor.chameleon_ori_image_tokenizer._vq_model.encode(batch)
    img_toks = img_toks.reshape(len(cropped), -1)
    image_toks = item_processor.chameleon_ori_translation.convert_img2bp2(img_toks)

    image_start_id = item_processor.token2id(item_processor.image_start_token)
    image_end_id = item_processor.token2id(item_processor.image_end_token)
    new_line_id = item_processor.token2id(item_processor.new_line_token)
    result = []
    for image, toks in zip(cropped, image_toks):
        w_grids = image.size[0] // item_processor.patch_size
        h_grids = image.size[1] // item_processor.patch_size
        h_latent = image.size[1] // 16
        w_latent = image.size[0] // 16
        full_image_toks = toks.reshape(h_latent, w_latent)
        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.full(
                    (h_latent, 1),
                    new_line_id,
                    device=full_image_toks.device,
                    dtype=full_image_toks.dtype,
                ),
            ),
            dim=1,
        ).flatten()
        result_toks = [
            image_start_id,
            item_processor.token2id(item_processor.get_n_grids_token(h_grids)),
            item_processor.token2id(item_processor.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            image_end_id,
        ]
        result.append({"input_ids": result_toks, "labels": result_toks})
    return result


def _bair_conversations(dataset: BairRobotPushingConversation, task_type: str) -> list[dict]:
    if task_type == "world":
        return [
            {
                "from": "human",
                "value": "Generate the next image based on the current image and action."
                + "<|image|><|action|>"
                + dataset.transition_prompt(),
            },
            {
                "from": "gpt",
                "value": "<|image|>",
            },
        ]
    if task_type == "action":
        return [
            {
                "from": "human",
                "value": "What action caused the transition between these two images?"
                + "<|image|><|image|>",
            },
            {
                "from": "gpt",
                "value": "<|action|>",
            },
        ]
    raise ValueError(f"unknown BAIR task type: {task_type}")


def _build_flatten_template(item_processor: FlexARItemProcessor_Action, conversations: list[dict]):
    d_media = {
        "<|image|>": [None, None],
        "<|action|>": [None],
    }
    source = item_processor.insert_implicit_media_symbol_in_q1(conversations, d_media)
    conversation, pieces = item_processor.add_speaker_and_signal(source)
    tokens = item_processor.tokenizer.encode(conversation, bos=True, eos=False)
    labels = [-100 for _ in tokens]

    for media_symbol, media_items in d_media.items():
        media_token = item_processor.d_media_symbol2token[media_symbol]
        media_token_count = tokens.count(media_token)
        if media_token_count != len(media_items):
            raise ValueError(
                f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
                f"but {len(media_items)} media placeholders are expected"
            )

    check_pos = 0
    for i, piece in enumerate(pieces):
        if i == 0:
            tokenized_value = item_processor.tokenizer.encode(piece["data"], bos=True, eos=False)
        else:
            tokenized_value = item_processor.tokenizer.encode_wo_prefix_space(piece["data"])
        if tokens[check_pos : check_pos + len(tokenized_value)] != tokenized_value:
            raise ValueError("inconsistent complete conversation and corresponding piece after tokenization")
        if piece["predict"]:
            labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value
        check_pos += len(tokenized_value)
    return tokens, labels


def _flatten_from_template(
    item_processor: FlexARItemProcessor_Action,
    template_tokens: list[int],
    template_labels: list[int],
    image_media: list[dict],
    action_media: list[dict],
):
    image_token = item_processor.d_media_symbol2token["<|image|>"]
    action_token = item_processor.d_media_symbol2token["<|action|>"]
    image_idx = 0
    action_idx = 0
    input_tokens = []
    labels = []
    for token, label in zip(template_tokens, template_labels):
        if token == image_token:
            media = image_media[image_idx]
            image_idx += 1
            input_tokens += media["input_ids"]
            labels += media["labels"] if label > 0 else [-100] * len(media["input_ids"])
        elif token == action_token:
            media = action_media[action_idx]
            action_idx += 1
            input_tokens += media["input_ids"]
            labels += media["labels"] if label > 0 else [-100] * len(media["input_ids"])
        else:
            input_tokens.append(token)
            labels.append(label)

    if image_idx != len(image_media) or action_idx != len(action_media):
        raise ValueError("media template replacement count mismatch")
    return input_tokens, labels


def _pretokenize_rank(args: argparse.Namespace) -> None:
    dataset = _build_dataset(args)
    item_processor = FlexARItemProcessor_Action(
        target_size=args.target_size,
        tokenizer=args.tokenizer,
    )
    templates = {
        task_type: _build_flatten_template(item_processor, _bair_conversations(dataset, task_type))
        for task_type in ("world", "action")
    }
    image_cache = {}

    def make_item(idx: int):
        task_type, file_idx, seq_idx, start, end = dataset.unpack_ref(dataset.data_list[idx])
        shard = dataset._load_shard(file_idx)
        frames = shard[dataset.frames_key][seq_idx]
        actions = shard[dataset.actions_key][seq_idx]

        image_i = dataset._frame_to_image(frames[start])
        image_j = dataset._frame_to_image(frames[end])
        image_i._bair_cache_key = (file_idx, seq_idx, start)
        image_j._bair_cache_key = (file_idx, seq_idx, end)
        action = actions[start:end].sum(axis=0).astype(np.float32)
        return task_type, [image_i, image_j], [action]

    def prime_image_cache(current_idx: int) -> None:
        if args.image_batch_size <= 1:
            return
        _, file_idx, seq_idx, start, frame_end = dataset.unpack_ref(dataset.data_list[current_idx])
        current_keys = ((file_idx, seq_idx, start), (file_idx, seq_idx, frame_end))
        if all(key in image_cache for key in current_keys):
            return

        keys = []
        images = []
        seen = set()
        pos = current_idx
        while pos < end and len(images) < args.image_batch_size:
            _, file_idx, seq_idx, start, frame_end = dataset.unpack_ref(dataset.data_list[pos])
            shard = dataset._load_shard(file_idx)
            frames = shard[dataset.frames_key][seq_idx]
            for frame_idx in (start, frame_end):
                key = (file_idx, seq_idx, frame_idx)
                if key in image_cache or key in seen:
                    continue
                image = dataset._frame_to_image(frames[frame_idx])
                image._bair_cache_key = key
                keys.append(key)
                images.append(image)
                seen.add(key)
                if len(images) >= args.image_batch_size:
                    break
            pos += 1

        for key, media in zip(keys, _process_images_batched(item_processor, images)):
            image_cache[key] = media
        while len(image_cache) > args.image_cache_size:
            image_cache.pop(next(iter(image_cache)))

    def cached_process_image(image):
        key = getattr(image, "_bair_cache_key", None)
        if key is not None and key in image_cache:
            return _copy_media(image_cache[key])
        media = item_processor.process_image(image)
        if key is not None:
            image_cache[key] = media
            while len(image_cache) > args.image_cache_size:
                image_cache.pop(next(iter(image_cache)))
        return _copy_media(media)

    item_processor.transform["<|image|>"] = cached_process_image

    out_dir = Path(args.out_dir)
    files_dir = out_dir / "files"
    shards_dir = out_dir / "shards"
    if args.storage_format == "pkl":
        files_dir.mkdir(parents=True, exist_ok=True)
    else:
        shards_dir.mkdir(parents=True, exist_ok=True)

    record_path = out_dir / f"{args.rank}-of-{args.splits}-record.jsonl"
    progress_path = out_dir / f"{args.rank}-of-{args.splits}-progress.txt"
    error_path = out_dir / f"{args.rank}-of-{args.splits}-errors.jsonl"
    done_ids = _load_done_ids(record_path)
    token_buffer = []
    label_buffer = []
    id_buffer = []
    record_buffer = []
    existing_shard_indices = []
    for path in shards_dir.glob(f"rank{args.rank:03d}_shard*_tokens.npy"):
        try:
            existing_shard_indices.append(int(path.stem.split("_shard", 1)[1].split("_", 1)[0]))
        except (IndexError, ValueError):
            pass
    shard_idx = max(existing_shard_indices, default=-1) + 1

    def flush_shard():
        nonlocal shard_idx
        if not token_buffer:
            return

        max_len = max(len(tokens) for tokens in token_buffer)
        tokens_arr = np.zeros((len(token_buffer), max_len), dtype=np.int32)
        labels_arr = np.full((len(label_buffer), max_len), -100, dtype=np.int32)
        for row, (tokens, labels) in enumerate(zip(token_buffer, label_buffer)):
            tokens_arr[row, : len(tokens)] = np.asarray(tokens, dtype=np.int32)
            labels_arr[row, : len(labels)] = np.asarray(labels, dtype=np.int32)

        stem = f"rank{args.rank:03d}_shard{shard_idx:06d}"
        token_path = (shards_dir / f"{stem}_tokens.npy").resolve()
        label_path = (shards_dir / f"{stem}_labels.npy").resolve()
        ids_path = (shards_dir / f"{stem}_ids.npy").resolve()
        np.save(token_path, tokens_arr)
        np.save(label_path, labels_arr)
        np.save(ids_path, np.asarray(id_buffer, dtype=np.int64))

        with record_path.open("a") as f:
            for local_idx, record in enumerate(record_buffer):
                record.update(
                    {
                        "token_file": str(token_path),
                        "label_file": str(label_path),
                        "ids_file": str(ids_path),
                        "index": local_idx,
                    }
                )
                f.write(json.dumps(record) + "\n")

        token_buffer.clear()
        label_buffer.clear()
        id_buffer.clear()
        record_buffer.clear()
        shard_idx += 1

    num_items = len(dataset)
    num_per_rank = math.ceil(num_items / args.splits)
    start = num_per_rank * args.rank
    end = min(num_per_rank * (args.rank + 1), num_items)
    print(f"rank {args.rank}/{args.splits}: processing [{start}, {end}) from {num_items} items")

    for idx in range(start, end):
        if idx in done_ids:
            continue
        should_log_progress = (idx - start) % args.log_every == 0 or idx == end - 1
        if should_log_progress:
            print(f"rank {args.rank}: {idx}/{end}")

        try:
            prime_image_cache(idx)
            task_type, images, actions = make_item(idx)
            image_media = [cached_process_image(image) for image in images]
            action_media = [item_processor.process_action(action) for action in actions]
            template_tokens, template_labels = templates[task_type]
            tokens, labels = _flatten_from_template(
                item_processor,
                template_tokens,
                template_labels,
                image_media,
                action_media,
            )
            if len(tokens) != len(labels):
                raise ValueError(f"token/label length mismatch: {len(tokens)} vs {len(labels)}")

            if args.storage_format == "pkl":
                pkl_path = (files_dir / f"{idx}.pkl").resolve()
                with pkl_path.open("wb") as f:
                    pickle.dump({"token": tokens, "label": labels, "id": idx}, f)
                record = {"file": str(pkl_path), "len": len(tokens), "id": idx, "task_type": task_type}
                with record_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            else:
                token_buffer.append(tokens)
                label_buffer.append(labels)
                id_buffer.append(idx)
                record_buffer.append({"len": len(tokens), "id": idx, "task_type": task_type})
                if len(token_buffer) >= args.shard_size:
                    flush_shard()
            done_ids.add(idx)
        except Exception:
            with error_path.open("a") as f:
                f.write(json.dumps({"id": idx, "error": format_exc()}) + "\n")

        if should_log_progress:
            with progress_path.open("w") as f:
                f.write("finished" if idx == end - 1 else str(idx))

    if start >= end:
        progress_path.write_text("finished")
    flush_shard()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize BAIR world-model samples for RynnVLA training.")
    parser.add_argument("--config", type=str, default="../configs/bair_robot_pushing/his_1_world_model_nopretokenize_train.yaml")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="../ckpts/base_model")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--storage-format", choices=["npy", "pkl"], default="npy")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--image-cache-size", type=int, default=8192)
    parser.add_argument("--with-transition-tokens", action="store_true")
    parser.add_argument("--transition-token-id", type=int, default=16001)
    parser.add_argument("--transition-token-count", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--save-name", type=str, default="record.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.aggregate_only:
        _aggregate_records(Path(args.out_dir), args.save_name)
    else:
        _pretokenize_rank(args)
