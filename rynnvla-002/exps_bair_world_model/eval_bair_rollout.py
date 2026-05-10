from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


SCRIPT_DIR = Path(__file__).resolve().parent
RYNNVLA_DIR = SCRIPT_DIR.parent
REPO_DIR = RYNNVLA_DIR.parent
BETA_DIR = REPO_DIR.parent.parent

for path in (str(RYNNVLA_DIR), str(REPO_DIR), str(BETA_DIR / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.dataset import reserved_token_from_id  # noqa: E402
from data.item_processor import FlexARItemProcessor_Action  # noqa: E402
from model import ChameleonXLLMXForConditionalGeneration_ck_action_head  # noqa: E402
from transition_field.metrics import (  # noqa: E402
    LPIPSMetric,
    TorchScriptFVDConfig,
    TorchScriptFVDFeatureExtractor,
    fvd_from_videos,
    video_quality_metrics,
)
from transition_field.visualize import save_prediction_gifs  # noqa: E402


IMAGE_START_ID = 8197
IMAGE_END_ID = 8196
DEFAULT_BASE_CHECKPOINT_PATH = "../ckpts/starting_point"
GENERATION_INSTRUCTION = "Generate the next image based on the current image and action."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline BAIR rollout eval for RynnVLA world models.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--base-checkpoint-path", type=str, default=DEFAULT_BASE_CHECKPOINT_PATH)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../../../../data/bair_robot_pushing_npz/test")
    parser.add_argument("--frames-key", type=str, default="frames")
    parser.add_argument("--actions-key", type=str, default="actions")
    parser.add_argument("--tokenizer-path", type=str, default="../ckpts/base_model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--future-frames", type=int, default=15)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--with-transition-tokens", action="store_true")
    parser.add_argument("--transition-token-id", type=int, default=16001)
    parser.add_argument("--transition-token-count", type=int, default=4)
    parser.add_argument("--transition-token-hidden-mult", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--time-horizon", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--z-loss-weight", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--rollout-state", choices=["token", "image"], default="token")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-use-cache", action="store_false", dest="use_cache")
    parser.add_argument("--eos-token-ids", type=str, default="8196,8710")
    parser.add_argument("--force-image-prefix", action="store_true", default=True)
    parser.add_argument("--no-force-image-prefix", action="store_false", dest="force_image_prefix")
    parser.add_argument(
        "--action-mode",
        choices=["gt", "zero", "shuffle", "negate_xy", "scale_xy"],
        default="gt",
        help="Action perturbation for action-sensitivity diagnostics. Default keeps ground-truth actions.",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--save-gif", action="store_true", default=True)
    parser.add_argument("--no-save-gif", action="store_false", dest="save_gif")
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--per-frame-horizons", type=str, default="1,5,10,15")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-device", type=str, default="")
    parser.add_argument("--fvd-model-path", type=str, default="")
    parser.add_argument("--fvd-samples", type=int, default=0)
    parser.add_argument("--fvd-batch-size", type=int, default=16)
    parser.add_argument("--fvd-resize-size", type=int, default=224)
    parser.add_argument("--fvd-min-frames", type=int, default=16)
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument(
        "--allow-nonformal-baseline",
        action="store_true",
        help="Allow a run label containing 'baseline' to use a smoke/non-formal checkpoint.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (SCRIPT_DIR / value).resolve()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = torch.device(device)
    if out.type == "cuda":
        torch.cuda.set_device(out)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_token_budget(resolution: int) -> int:
    latent = resolution // 16
    return latent * (latent + 1) + 4


def image_remainder_budget(resolution: int) -> int:
    latent = resolution // 16
    return latent * (latent + 1) + 1


def default_max_new_tokens(args: argparse.Namespace) -> int:
    if args.max_new_tokens > 0:
        return int(args.max_new_tokens)
    if args.force_image_prefix:
        return image_remainder_budget(args.resolution)
    return image_token_budget(args.resolution) + 16


def image_prefix_tokens(item_processor: FlexARItemProcessor_Action, resolution: int) -> list[int]:
    grids = resolution // item_processor.patch_size
    return [
        item_processor.token2id(item_processor.image_start_token),
        item_processor.token2id(item_processor.get_n_grids_token(grids)),
        item_processor.token2id(item_processor.get_n_grids_token(grids)),
    ]


def parse_eos_token_ids(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def transition_prompt(args: argparse.Namespace) -> str:
    if not args.with_transition_tokens or args.transition_token_count <= 0:
        return ""
    return reserved_token_from_id(args.transition_token_id) * args.transition_token_count


def generation_prompt(args: argparse.Namespace) -> str:
    return GENERATION_INSTRUCTION + "<|image|><|action|>" + transition_prompt(args)


def load_checkpoint_meta(checkpoint_path: Path) -> dict[str, object]:
    meta_path = checkpoint_path / "checkpoint_meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def adapter_base_checkpoint_path(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    explicit_base = args.base_checkpoint_path != DEFAULT_BASE_CHECKPOINT_PATH
    if explicit_base:
        return resolve_path(args.base_checkpoint_path)
    meta = load_checkpoint_meta(checkpoint_path)
    base_model_path = str(meta.get("base_model_path", ""))
    if base_model_path:
        return resolve_path(base_model_path)
    return resolve_path(args.base_checkpoint_path)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    checkpoint_path = resolve_path(args.checkpoint_path)
    trainable_path = checkpoint_path / "trainable_params.pt"
    model_path = adapter_base_checkpoint_path(args, checkpoint_path) if trainable_path.exists() else checkpoint_path
    model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        str(model_path),
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        transition_token_id=args.transition_token_id,
        transition_token_count=args.transition_token_count,
        transition_token_hidden_mult=args.transition_token_hidden_mult,
        max_position_embeddings=args.max_seq_len,
        mask_image_logits=False,
        dropout=args.dropout,
        z_loss_weight=args.z_loss_weight,
        torch_dtype=dtype,
        device_map="cpu",
    )
    if trainable_path.exists():
        model.transition_token_adapter.reset_stable_parameters()
        trainable_state = torch.load(trainable_path, map_location="cpu")
        incompatible = model.load_state_dict(trainable_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_trainable = [
            key
            for key, _ in model.named_parameters()
            if key.startswith("transition_token_adapter.") and key in incompatible.missing_keys
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                f"failed to load trainable checkpoint {trainable_path}: "
                f"unexpected={unexpected}, missing_trainable={missing_trainable}"
            )
    if hasattr(model.model, "vqmodel"):
        del model.model.vqmodel
    model.config.use_cache = bool(args.use_cache)
    model.to(device)
    model.eval()
    return model


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def _checkpoint_has_full_weights(path: Path) -> bool:
    if (path / "model.safetensors.index.json").exists() or (path / "pytorch_model.bin.index.json").exists():
        return True
    if any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin")):
        return True
    return False


def checkpoint_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = resolve_path(args.checkpoint_path)
    base_checkpoint_path = resolve_path(args.base_checkpoint_path)
    trainable_path = checkpoint_path / "trainable_params.pt"
    meta = load_checkpoint_meta(checkpoint_path)
    base_model_path = str(meta.get("base_model_path", ""))
    checkpoint_format = str(meta.get("checkpoint_format", ""))

    is_starting_point = _same_resolved_path(checkpoint_path, base_checkpoint_path)
    has_adapter = trainable_path.exists()
    resolved_adapter_base = adapter_base_checkpoint_path(args, checkpoint_path) if has_adapter else base_checkpoint_path
    has_full_weights = _checkpoint_has_full_weights(checkpoint_path)
    base_is_starting_point = "starting_point" in str(resolved_adapter_base)

    if is_starting_point and not has_adapter:
        kind = "starting_point_smoke"
        formal_rollout_candidate = False
        warning = (
            "This is the raw Chameleon/RynnVLA starting point, not a BAIR-finetuned "
            "world-model checkpoint. Use it only for rollout plumbing smoke tests."
        )
    elif has_adapter and base_is_starting_point:
        kind = "transition_adapter_on_starting_point"
        formal_rollout_candidate = False
        warning = (
            "This checkpoint only contains a transition adapter on top of the raw "
            "starting point. It validates training/checkpoint plumbing, but it is not "
            "a formal BAIR video-generation baseline or final candidate."
        )
    elif has_adapter:
        kind = "transition_adapter"
        formal_rollout_candidate = True
        warning = ""
    elif has_full_weights:
        kind = "full_model_checkpoint"
        formal_rollout_candidate = True
        warning = ""
    else:
        kind = "unknown"
        formal_rollout_candidate = False
        warning = "Checkpoint format is unknown; do not use this as a formal rollout comparison."

    return {
        "checkpoint_kind": kind,
        "checkpoint_format": checkpoint_format,
        "has_trainable_params": has_adapter,
        "has_full_weights": has_full_weights,
        "base_model_path": base_model_path,
        "resolved_base_checkpoint_path": str(resolved_adapter_base),
        "formal_rollout_candidate": formal_rollout_candidate,
        "warning": warning,
    }


def validate_rollout_target(args: argparse.Namespace, diagnostics: dict[str, object]) -> None:
    label = args.run_label.lower()
    wants_baseline = "baseline" in label
    smoke_run = "smoke" in label
    formal = bool(diagnostics.get("formal_rollout_candidate", False))
    if diagnostics.get("warning"):
        print(f"[eval:warning] {diagnostics['warning']}", flush=True)
    if wants_baseline and not formal and not smoke_run and not args.allow_nonformal_baseline:
        raise ValueError(
            "run label contains 'baseline', but the checkpoint is not a formal BAIR-finetuned "
            f"baseline ({diagnostics.get('checkpoint_kind')}). Add 'smoke' to the run label for "
            "a plumbing test, or pass --allow-nonformal-baseline explicitly."
        )


class ImageBlockLogitsProcessor(LogitsProcessor):
    """Constrain generation inside a Chameleon image block.

    The prompt may already contain `<image_start>, h_grid, w_grid`. Once an
    image block is open, image positions are restricted to VQ image tokens,
    row boundaries are forced to newline, and the final position is forced to
    image end.
    """

    def __init__(
        self,
        *,
        image_token_ids: list[int],
        image_start_token_id: int,
        image_end_token_id: int,
        image_next_line_token_id: int,
    ) -> None:
        self.image_token_ids = image_token_ids
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index: int | None = None
        self.h_latent_dim: int | None = None
        self.w_latent_dim: int | None = None

    def _force_token(self, scores: torch.FloatTensor, token_id: int) -> torch.FloatTensor:
        constrained = torch.full_like(scores, -math.inf)
        constrained[:, token_id] = 0.0
        return constrained

    def _allow_image_tokens(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        constrained = torch.full_like(scores, -math.inf)
        image_token_ids = torch.tensor(self.image_token_ids, device=scores.device, dtype=torch.long)
        constrained[:, image_token_ids] = scores[:, image_token_ids]
        return constrained

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        image_starts = int((input_ids[0] == self.image_start_token_id).sum().item())
        image_ends = int((input_ids[0] == self.image_end_token_id).sum().item())
        if image_starts == image_ends:
            self.image_start_token_id_index = None
            self.h_latent_dim = None
            self.w_latent_dim = None
            return scores
        if image_starts != image_ends + 1:
            return scores

        if self.image_start_token_id_index is None:
            self.image_start_token_id_index = int(torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item())
        new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
        if new_token_num < 2:
            return scores

        if self.h_latent_dim is None or self.w_latent_dim is None:
            h_grids = int(input_ids[0][self.image_start_token_id_index + 1].item()) - 8804
            w_grids = int(input_ids[0][self.image_start_token_id_index + 2].item()) - 8804
            self.h_latent_dim = h_grids * 2
            self.w_latent_dim = w_grids * 2

        tokens_after_grid = input_ids[0][self.image_start_token_id_index + 3 :]
        next_pos = len(tokens_after_grid) + 1
        row_span = self.w_latent_dim + 1
        total_content = row_span * self.h_latent_dim
        if next_pos == total_content + 1:
            return self._force_token(scores, self.image_end_token_id)
        if next_pos % row_span == 0:
            return self._force_token(scores, self.image_next_line_token_id)
        return self._allow_image_tokens(scores)


def build_logits_processor(
    model: torch.nn.Module,
    item_processor: FlexARItemProcessor_Action,
) -> LogitsProcessorList:
    image_tokens = list(getattr(model.model.vocabulary_mapping, "image_tokens"))
    return LogitsProcessorList(
        [
            ImageBlockLogitsProcessor(
                image_token_ids=image_tokens,
                image_start_token_id=item_processor.token2id(item_processor.image_start_token),
                image_end_token_id=item_processor.token2id(item_processor.image_end_token),
                image_next_line_token_id=item_processor.token2id(item_processor.new_line_token),
            )
        ]
    )


def scan_sequence_refs(
    data_dir: Path,
    *,
    frames_key: str,
    actions_key: str,
    context_frames: int,
    future_frames: int,
    max_shards: int,
) -> tuple[list[Path], list[tuple[int, int]]]:
    files = sorted(data_dir.glob("*.npz"))
    if max_shards > 0:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"no .npz shards found under {data_dir}")

    refs: list[tuple[int, int]] = []
    min_frames = context_frames + future_frames
    min_actions = context_frames + future_frames - 1
    for file_idx, shard_path in enumerate(files):
        with np.load(shard_path) as shard:
            frames_shape = shard[frames_key].shape
            actions_shape = shard[actions_key].shape
        if len(frames_shape) != 5:
            raise ValueError(f"{shard_path} {frames_key} must be [N,T,H,W,C], got {frames_shape}")
        if frames_shape[1] < min_frames or actions_shape[1] < min_actions:
            continue
        refs.extend((file_idx, seq_idx) for seq_idx in range(frames_shape[0]))
    if not refs:
        raise ValueError(
            f"no sequences have enough frames/actions for context={context_frames}, future={future_frames}"
        )
    return files, refs


def select_refs(refs: list[tuple[int, int]], samples: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    if samples <= len(refs):
        return rng.sample(refs, samples)
    return [rng.choice(refs) for _ in range(samples)]


def iter_selected_sequences(
    files: list[Path],
    selected_entries: list[tuple[int, tuple[int, int]]],
    *,
    frames_key: str,
    actions_key: str,
) -> Iterable[tuple[int, np.ndarray, np.ndarray, Path, int]]:
    refs_by_file: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sample_idx, (file_idx, seq_idx) in selected_entries:
        refs_by_file[file_idx].append((sample_idx, seq_idx))

    for file_idx in sorted(refs_by_file):
        shard_path = files[file_idx]
        with np.load(shard_path) as shard:
            frames = shard[frames_key]
            actions = shard[actions_key]
            for sample_idx, seq_idx in refs_by_file[file_idx]:
                yield sample_idx, frames[seq_idx], actions[seq_idx], shard_path, seq_idx


def rollout_action_indices(
    *,
    action_start: int,
    future_frames: int,
    sample_idx: int,
    seed: int,
    action_mode: str,
) -> list[int]:
    indices = list(range(action_start, action_start + future_frames))
    if action_mode != "shuffle":
        return indices
    rng = random.Random(seed * 100_003 + sample_idx)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    return shuffled


def transform_action(action: np.ndarray, *, args: argparse.Namespace) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if args.action_mode in {"gt", "shuffle"}:
        return out
    if args.action_mode == "zero":
        return np.zeros_like(out)
    if args.action_mode == "negate_xy":
        out[..., :2] *= -1.0
        return out
    if args.action_mode == "scale_xy":
        out[..., :2] *= float(args.action_scale)
        return out
    raise ValueError(f"unsupported action_mode={args.action_mode}")


def frame_to_pil(frame: np.ndarray, resolution: int) -> Image.Image:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"expected frame [H,W,C] or [C,H,W], got {frame.shape}")
    if frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    image = Image.fromarray(frame).convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BICUBIC)
    return image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_generation_tokens(
    image: Image.Image,
    action: np.ndarray,
    *,
    item_processor: FlexARItemProcessor_Action,
    args: argparse.Namespace,
) -> list[int]:
    conv = {
        "conversations": [
            {
                "from": "human",
                "value": generation_prompt(args),
            }
        ],
        "image": [image],
        "action": [np.asarray(action, dtype=np.float32)],
    }
    return item_processor.process_item(conv, training_mode=False)


def encode_image_tokens(image: Image.Image, item_processor: FlexARItemProcessor_Action) -> list[int]:
    return [int(token) for token in item_processor.process_image(image)["input_ids"]]


def build_generation_tokens_from_image_tokens(
    image_tokens: list[int],
    action: np.ndarray,
    *,
    item_processor: FlexARItemProcessor_Action,
    args: argparse.Namespace,
) -> list[int]:
    conversation, _ = item_processor.add_speaker_and_signal(
        [{"from": "human", "value": generation_prompt(args)}]
    )
    tokens = item_processor.tokenizer.encode(conversation, bos=True, eos=False)
    image_placeholder = item_processor.d_media_symbol2token["<|image|>"]
    action_placeholder = item_processor.d_media_symbol2token["<|action|>"]
    if tokens.count(image_placeholder) != 1 or tokens.count(action_placeholder) != 1:
        raise ValueError(
            "generation prompt must contain exactly one image placeholder and one action placeholder"
        )

    action_tokens = [
        int(token)
        for token in item_processor.process_action(np.asarray(action, dtype=np.float32))["input_ids"]
    ]
    output: list[int] = []
    for token in tokens:
        if token == image_placeholder:
            output.extend(int(value) for value in image_tokens)
        elif token == action_placeholder:
            output.extend(action_tokens)
        else:
            output.append(int(token))
    return output


def extract_first_image_tokens(tokens: list[int]) -> list[int]:
    start_idx = None
    for idx, token in enumerate(tokens):
        if token == IMAGE_START_ID:
            start_idx = idx
            break
    if start_idx is None:
        if len(tokens) >= 4 and 8804 <= tokens[0] <= 9000 and 8804 <= tokens[1] <= 9000:
            return tokens
        raise ValueError("generated sequence does not contain an image start token")

    end_idx = None
    for idx in range(start_idx + 1, len(tokens)):
        if tokens[idx] == IMAGE_END_ID:
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError("generated sequence does not contain a complete image block")
    return tokens[start_idx : end_idx + 1]


def generate_image_from_prompt_tokens(
    model: torch.nn.Module,
    item_processor: FlexARItemProcessor_Action,
    tokens: list[int],
    args: argparse.Namespace,
    device: torch.device,
    generation_config: GenerationConfig,
    *,
    fallback_image_tokens: list[int] | None,
    fallback_image: Image.Image,
) -> tuple[list[int], Image.Image, dict[str, float | str]]:
    forced_prefix: list[int] = []
    if args.force_image_prefix:
        forced_prefix = image_prefix_tokens(item_processor, args.resolution)
        tokens = tokens + forced_prefix
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
    meta: dict[str, float | str] = {
        "prompt_tokens": float(input_ids.shape[1]),
    }
    raw_tokens: list[int] = []
    try:
        if hasattr(model, "init_input_ids"):
            model.init_input_ids = None
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.amp and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            if args.force_image_prefix:
                result = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    generation_config=generation_config,
                    logits_processor=build_logits_processor(model, item_processor),
                    output_hidden_states=True,
                    training=False,
                    return_dict_in_generate=True,
                    att_mask=None,
                )
                generated = result["sequences"][:, input_ids.shape[1] :]
            else:
                generated = model.generate_img(input_ids, generation_config)
        raw_tokens = generated[0].detach().cpu().tolist()
        if forced_prefix:
            image_tokens = forced_prefix + raw_tokens
            if IMAGE_END_ID in image_tokens:
                image_tokens = image_tokens[: image_tokens.index(IMAGE_END_ID) + 1]
        else:
            image_tokens = extract_first_image_tokens(raw_tokens)
        image = item_processor.decode_image(list(image_tokens)).convert("RGB")
        if image.size != (args.resolution, args.resolution):
            image = image.resize((args.resolution, args.resolution), Image.BICUBIC)
        meta.update(
            {
                "generated_tokens": float(len(raw_tokens)),
                "image_tokens": float(len(image_tokens)),
                "decode_failed": 0.0,
            }
        )
        return [int(token) for token in image_tokens], image, meta
    except Exception as exc:  # noqa: BLE001
        meta.update(
            {
                "generated_tokens": 0.0,
                "image_tokens": 0.0,
                "decode_failed": 1.0,
                "failure_reason": str(exc)[:200],
            }
        )
        if raw_tokens:
            meta["token_preview"] = ",".join(str(token) for token in raw_tokens[:32])
        return list(fallback_image_tokens or []), fallback_image.copy(), meta


def generate_next_image(
    model: torch.nn.Module,
    item_processor: FlexARItemProcessor_Action,
    current_image: Image.Image,
    action: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    generation_config: GenerationConfig,
) -> tuple[Image.Image, dict[str, float | str]]:
    tokens = build_generation_tokens(current_image, action, item_processor=item_processor, args=args)
    _, image, meta = generate_image_from_prompt_tokens(
        model,
        item_processor,
        tokens,
        args,
        device,
        generation_config,
        fallback_image_tokens=None,
        fallback_image=current_image,
    )
    return image, meta


def generate_next_image_tokens(
    model: torch.nn.Module,
    item_processor: FlexARItemProcessor_Action,
    current_image_tokens: list[int],
    current_image: Image.Image,
    action: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    generation_config: GenerationConfig,
) -> tuple[list[int], Image.Image, dict[str, float | str]]:
    tokens = build_generation_tokens_from_image_tokens(
        current_image_tokens,
        action,
        item_processor=item_processor,
        args=args,
    )
    return generate_image_from_prompt_tokens(
        model,
        item_processor,
        tokens,
        args,
        device,
        generation_config,
        fallback_image_tokens=current_image_tokens,
        fallback_image=current_image,
    )


def collect_rollouts(
    model: torch.nn.Module,
    item_processor: FlexARItemProcessor_Action,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int | str]]]:
    data_dir = resolve_path(args.data_dir)
    files, refs = scan_sequence_refs(
        data_dir,
        frames_key=args.frames_key,
        actions_key=args.actions_key,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        max_shards=args.max_shards,
    )
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    selected_all = select_refs(refs, args.samples, args.seed)
    selected = [
        (sample_idx, ref)
        for sample_idx, ref in enumerate(selected_all)
        if sample_idx % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError(
            f"rollout shard {args.shard_index}/{args.shard_count} has no samples "
            f"from requested samples={args.samples}"
        )
    max_new_tokens = default_max_new_tokens(args)
    eos_token_ids = parse_eos_token_ids(args.eos_token_ids)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        max_length=model.config.max_position_embeddings,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        do_sample=args.do_sample,
        eos_token_id=eos_token_ids,
        pad_token_id=eos_token_ids[-1] if eos_token_ids else None,
        use_cache=args.use_cache,
    )

    gt_by_idx: dict[int, torch.Tensor] = {}
    pred_by_idx: dict[int, torch.Tensor] = {}
    sample_meta: list[dict[str, float | int | str]] = []
    start_time = time.time()
    for ordinal, (sample_idx, frames, actions, shard_path, seq_idx) in enumerate(
        iter_selected_sequences(
            files,
            selected,
            frames_key=args.frames_key,
            actions_key=args.actions_key,
        )
    ):
        gt_images = [frame_to_pil(frame, args.resolution) for frame in frames[: args.context_frames + args.future_frames]]
        generated_images = list(gt_images[: args.context_frames])
        current_image = generated_images[-1]
        current_image_tokens = encode_image_tokens(current_image, item_processor) if args.rollout_state == "token" else []
        action_start = args.context_frames - 1
        action_indices = rollout_action_indices(
            action_start=action_start,
            future_frames=args.future_frames,
            sample_idx=sample_idx,
            seed=args.seed,
            action_mode=args.action_mode,
        )
        failures = 0
        token_lengths: list[float] = []
        prompt_lengths: list[float] = []
        first_failure = ""
        first_failure_tokens = ""
        for step in range(args.future_frames):
            action = transform_action(actions[action_indices[step]], args=args)
            if args.rollout_state == "token":
                current_image_tokens, current_image, meta = generate_next_image_tokens(
                    model,
                    item_processor,
                    current_image_tokens,
                    current_image,
                    action,
                    args,
                    device,
                    generation_config,
                )
            else:
                current_image, meta = generate_next_image(
                    model,
                    item_processor,
                    current_image,
                    action,
                    args,
                    device,
                    generation_config,
                )
            generated_images.append(current_image)
            failures += int(float(meta.get("decode_failed", 0.0)) > 0)
            token_lengths.append(float(meta.get("image_tokens", 0.0)))
            prompt_lengths.append(float(meta.get("prompt_tokens", 0.0)))
            if not first_failure and "failure_reason" in meta:
                first_failure = str(meta["failure_reason"])
                first_failure_tokens = str(meta.get("token_preview", ""))

        gt_by_idx[sample_idx] = torch.stack([pil_to_tensor(image) for image in gt_images], dim=0)
        pred_by_idx[sample_idx] = torch.stack([pil_to_tensor(image) for image in generated_images], dim=0)
        sample_meta.append(
            {
                "sample_idx": sample_idx,
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "shard": str(shard_path),
                "seq_idx": int(seq_idx),
                "decode_failures": failures,
                "mean_image_tokens": float(np.mean(token_lengths)) if token_lengths else 0.0,
                "mean_prompt_tokens": float(np.mean(prompt_lengths)) if prompt_lengths else 0.0,
                "rollout_state": args.rollout_state,
                "action_mode": args.action_mode,
                "action_scale": float(args.action_scale),
                "first_failure": first_failure,
                "first_failure_tokens": first_failure_tokens,
            }
        )
        done = ordinal + 1
        elapsed = time.time() - start_time
        save_partial_artifacts(
            gt_by_idx,
            pred_by_idx,
            sample_meta,
            args,
            completed=done,
            total=len(selected),
            elapsed=elapsed,
            latest_sample_idx=sample_idx,
        )
        print(
            f"[rollout] shard {args.shard_index}/{args.shard_count} {done}/{len(selected)} clips, "
            f"{elapsed / max(done, 1):.2f}s/clip, failures={sum(int(m['decode_failures']) for m in sample_meta)}",
            flush=True,
        )

    return stack_available_clips(gt_by_idx, pred_by_idx), sample_meta


def parse_horizons(text: str, future_frames: int) -> list[int]:
    horizons: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if 1 <= value <= future_frames and value not in horizons:
            horizons.append(value)
    return horizons


def maybe_lpips_metric(args: argparse.Namespace, device: torch.device) -> LPIPSMetric | None:
    if not args.lpips:
        return None
    return LPIPSMetric(net=args.lpips_net, device=args.lpips_device or str(device))


def grouped_metrics(
    clips: dict[str, torch.Tensor],
    sample_meta: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    device: torch.device,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    gt = clips["gt"]
    pred = clips["rollout"]
    context = args.context_frames
    gt_future = gt[:, context:]
    pred_future = pred[:, context:]
    lpips_metric = maybe_lpips_metric(args, device)

    quality = {}
    quality.update(video_quality_metrics(pred_future, gt_future, prefix="future", lpips_metric=lpips_metric))
    quality.update(video_quality_metrics(pred, gt, prefix="clip", lpips_metric=lpips_metric))

    horizons = parse_horizons(args.per_frame_horizons, args.future_frames)
    per_horizon: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        idx = horizon - 1
        per_horizon[str(horizon)] = video_quality_metrics(
            pred_future[:, idx : idx + 1],
            gt_future[:, idx : idx + 1],
            prefix="frame",
            lpips_metric=lpips_metric,
        )

    fvd = {}
    if args.fvd_samples > 0:
        if args.fvd_samples < 2:
            raise ValueError("--fvd-samples must be at least 2")
        if args.fvd_samples > gt.shape[0]:
            raise ValueError("--fvd-samples cannot exceed --samples")
        if not args.fvd_model_path:
            raise ValueError("--fvd-model-path is required when --fvd-samples > 0")
        extractor = TorchScriptFVDFeatureExtractor(
            TorchScriptFVDConfig(
                model_path=str(resolve_path(args.fvd_model_path)),
                resize_size=args.fvd_resize_size,
                batch_size=args.fvd_batch_size,
                min_frames=args.fvd_min_frames,
            ),
            device=device,
        )
        fvd = {
            "rollout_fvd": float(fvd_from_videos(gt[: args.fvd_samples], pred[: args.fvd_samples], extractor)),
            "samples": float(args.fvd_samples),
            "model_path": str(resolve_path(args.fvd_model_path)),
            "resize_size": float(args.fvd_resize_size),
            "min_frames": float(args.fvd_min_frames),
        }

    total_failures = sum(int(meta["decode_failures"]) for meta in sample_meta)
    total_steps = int(gt_future.shape[0]) * args.future_frames
    failures = {
        "decode_failures": float(total_failures),
        "decode_failure_rate": float(total_failures / max(total_steps, 1)),
        "clips_with_failures": float(sum(int(meta["decode_failures"]) > 0 for meta in sample_meta)),
    }

    return {
        "run": {
            "label": args.run_label,
            "checkpoint_path": str(resolve_path(args.checkpoint_path)),
            "data_dir": str(resolve_path(args.data_dir)),
            "device": str(device),
            "seed": float(args.seed),
        },
        "config": {
            "samples": float(gt.shape[0]),
            "samples_requested": float(args.samples),
            "shard_index": float(args.shard_index),
            "shard_count": float(args.shard_count),
            "context_frames": float(args.context_frames),
            "future_frames": float(args.future_frames),
            "clip_frames": float(gt.shape[1]),
            "resolution": float(args.resolution),
            "with_transition_tokens": bool(args.with_transition_tokens),
            "transition_token_id": float(args.transition_token_id),
            "transition_token_count": float(args.transition_token_count),
            "max_new_tokens": float(default_max_new_tokens(args)),
            "rollout_state": args.rollout_state,
            "action_mode": args.action_mode,
            "action_scale": float(args.action_scale),
            "force_image_prefix": bool(args.force_image_prefix),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_k": float(args.top_k),
        },
        "quality": quality,
        "per_horizon": per_horizon,
        "fvd": fvd,
        "failures": failures,
        "diagnostics": diagnostics,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def stack_available_clips(
    gt_by_idx: dict[int, torch.Tensor],
    pred_by_idx: dict[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    indices = sorted(gt_by_idx)
    return {
        "gt": torch.stack([gt_by_idx[idx] for idx in indices], dim=0),
        "rollout": torch.stack([pred_by_idx[idx] for idx in indices], dim=0),
    }


def partial_rollout_metrics(
    clips: dict[str, torch.Tensor],
    sample_meta: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    *,
    completed: int,
    total: int,
    elapsed: float,
) -> dict[str, object]:
    gt = clips["gt"]
    pred = clips["rollout"]
    context = args.context_frames
    gt_future = gt[:, context:]
    pred_future = pred[:, context:]
    quality = {}
    quality.update(video_quality_metrics(pred_future, gt_future, prefix="future"))
    quality.update(video_quality_metrics(pred, gt, prefix="clip"))

    horizons = parse_horizons(args.per_frame_horizons, args.future_frames)
    per_horizon: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        idx = horizon - 1
        per_horizon[str(horizon)] = video_quality_metrics(
            pred_future[:, idx : idx + 1],
            gt_future[:, idx : idx + 1],
            prefix="frame",
        )

    total_failures = sum(int(meta["decode_failures"]) for meta in sample_meta)
    total_steps = max(completed, 1) * args.future_frames
    return {
        "run": {
            "label": args.run_label,
            "checkpoint_path": str(resolve_path(args.checkpoint_path)),
            "data_dir": str(resolve_path(args.data_dir)),
            "status": "partial",
        },
        "progress": {
            "completed": float(completed),
            "total": float(total),
            "elapsed_sec": float(elapsed),
            "sec_per_clip": float(elapsed / max(completed, 1)),
        },
        "config": {
            "samples_completed": float(completed),
            "samples_requested": float(args.samples),
            "shard_index": float(args.shard_index),
            "shard_count": float(args.shard_count),
            "context_frames": float(args.context_frames),
            "future_frames": float(args.future_frames),
            "clip_frames": float(gt.shape[1]),
            "resolution": float(args.resolution),
            "with_transition_tokens": bool(args.with_transition_tokens),
            "transition_token_count": float(args.transition_token_count),
            "rollout_state": args.rollout_state,
            "action_mode": args.action_mode,
            "action_scale": float(args.action_scale),
            "force_image_prefix": bool(args.force_image_prefix),
        },
        "quality": quality,
        "per_horizon": per_horizon,
        "failures": {
            "decode_failures": float(total_failures),
            "decode_failure_rate": float(total_failures / total_steps),
            "clips_with_failures": float(sum(int(meta["decode_failures"]) > 0 for meta in sample_meta)),
        },
    }


def save_partial_artifacts(
    gt_by_idx: dict[int, torch.Tensor],
    pred_by_idx: dict[int, torch.Tensor],
    sample_meta: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    *,
    completed: int,
    total: int,
    elapsed: float,
    latest_sample_idx: int,
) -> None:
    out_dir = resolve_path(args.out_dir)
    clips = stack_available_clips(gt_by_idx, pred_by_idx)
    save_json(out_dir / "partial_metrics.json", partial_rollout_metrics(clips, sample_meta, args, completed=completed, total=total, elapsed=elapsed))
    save_json(out_dir / "partial_sample_manifest.json", sample_meta)
    save_json(
        out_dir / "partial_status.json",
        {
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed,
            "sec_per_clip": elapsed / max(completed, 1),
            "latest_sample_idx": latest_sample_idx,
        },
    )
    if args.save_gif and completed <= args.preview_samples:
        save_prediction_gifs(
            {
                "gt": gt_by_idx[latest_sample_idx].unsqueeze(0),
                "rollout": pred_by_idx[latest_sample_idx].unsqueeze(0),
            },
            out_dir / "partial_rollout_videos",
            prefix=f"partial_clip_{latest_sample_idx:03d}",
            fps=args.video_fps,
            max_samples=1,
        )
    if completed <= min(args.preview_samples, 4):
        save_prediction_grid(clips, out_dir / "partial_prediction_grid.png", max_samples=min(args.preview_samples, 4))


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def labeled_tile(image: torch.Tensor, label: str, label_height: int = 18) -> Image.Image:
    tile = tensor_image_to_pil(image)
    out = Image.new("RGB", (tile.width, tile.height + label_height), "white")
    out.paste(tile, (0, label_height))
    draw = ImageDraw.Draw(out)
    draw.text((3, 2), label, fill=(0, 0, 0))
    return out


def save_prediction_grid(clips: dict[str, torch.Tensor], path: Path, *, max_samples: int = 4) -> None:
    gt = clips["gt"][:max_samples]
    pred = clips["rollout"][:max_samples]
    rows: list[Image.Image] = []
    for sample_idx in range(gt.shape[0]):
        tiles: list[Image.Image] = []
        for frame_idx in range(gt.shape[1]):
            tiles.append(labeled_tile(gt[sample_idx, frame_idx], f"gt t{frame_idx}"))
        for frame_idx in range(pred.shape[1]):
            tiles.append(labeled_tile(pred[sample_idx, frame_idx], f"rollout t{frame_idx}"))
        width = sum(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        row = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        rows.append(row)

    total_width = max(row.width for row in rows)
    total_height = sum(row.height for row in rows)
    grid = Image.new("RGB", (total_width, total_height), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def save_horizon_plot(metrics: dict[str, object], path: Path) -> None:
    per_horizon = metrics.get("per_horizon", {})
    if not isinstance(per_horizon, dict) or not per_horizon:
        return
    import matplotlib.pyplot as plt

    horizons = sorted(int(key) for key in per_horizon.keys())
    keys = ["frame_mse", "frame_psnr", "frame_ssim", "frame_lpips"]
    available = [
        key
        for key in keys
        if any(key in per_horizon[str(horizon)] for horizon in horizons)
    ]
    if not available:
        return
    cols = 2
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.0), squeeze=False)
    for ax, key in zip(axes.flat, available):
        values = [per_horizon[str(horizon)].get(key, float("nan")) for horizon in horizons]
        ax.plot(horizons, values, marker="o", linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("future horizon")
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(available) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_artifacts(
    clips: dict[str, torch.Tensor],
    sample_meta: list[dict[str, float | int | str]],
    metrics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "sample_manifest.json", sample_meta)
    save_json(out_dir / "run_config.json", vars(args))

    artifacts: dict[str, object] = {}
    save_prediction_grid(clips, out_dir / "prediction_grid.png", max_samples=min(args.preview_samples, 4))
    artifacts["prediction_grid"] = str(out_dir / "prediction_grid.png")
    save_horizon_plot(metrics, out_dir / "horizon_metrics.png")
    artifacts["horizon_metrics"] = str(out_dir / "horizon_metrics.png")
    if args.save_gif:
        gif_paths = save_prediction_gifs(
            {
                "gt": clips["gt"][: args.preview_samples],
                "rollout": clips["rollout"][: args.preview_samples],
            },
            out_dir / "rollout_videos",
            prefix="rollout_clip",
            fps=args.video_fps,
            max_samples=args.preview_samples,
        )
        artifacts["rollout_gifs"] = [str(path) for path in gif_paths]
    save_json(out_dir / "rollout_artifacts.json", artifacts)
    cleanup_partial_artifacts(out_dir)


def cleanup_partial_artifacts(out_dir: Path) -> None:
    for path in (
        out_dir / "partial_metrics.json",
        out_dir / "partial_sample_manifest.json",
        out_dir / "partial_status.json",
        out_dir / "partial_prediction_grid.png",
    ):
        path.unlink(missing_ok=True)
    shutil.rmtree(out_dir / "partial_rollout_videos", ignore_errors=True)


def main() -> None:
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[eval] device={device} checkpoint={resolve_path(args.checkpoint_path)}", flush=True)
    diagnostics = checkpoint_diagnostics(args)
    validate_rollout_target(args, diagnostics)
    item_processor = FlexARItemProcessor_Action(
        tokenizer=str(resolve_path(args.tokenizer_path)),
        target_size=args.resolution,
    )
    model = load_model(args, device)
    clips, sample_meta = collect_rollouts(model, item_processor, args, device)
    metrics = grouped_metrics(clips, sample_meta, args, device, diagnostics)
    save_artifacts(clips, sample_meta, metrics, args)
    print(f"[eval] wrote {resolve_path(args.out_dir)}", flush=True)


if __name__ == "__main__":
    main()
