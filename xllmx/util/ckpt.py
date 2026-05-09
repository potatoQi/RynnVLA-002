import json
import logging
import os
import shutil
from typing import Dict, Optional
import subprocess

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

logger = logging.getLogger(__name__)


def split_ckpt_str_into_epoch_iter(ckpt_str: str):
    # divide ckpt directory names into epoch and iter parts
    parts = ckpt_str.split("-")
    epoch = int(parts[0].replace("epoch", ""))
    if len(parts) == 2:
        iter_part = int(parts[1].replace("iter", ""))
    else:
        iter_part = None
    return epoch, iter_part


def remove_early_ckpts(out_dir, max_keep=2):

    if max_keep <= 0:
        return

    def ckpt_sort_key(s):
        # divide ckpt directory names into epoch and iter parts
        epoch, iteration = split_ckpt_str_into_epoch_iter(s)
        if iteration is None:
            iteration = float("inf")
        return epoch, iteration

    existing_checkpoints = [_ for _ in os.listdir(out_dir) if "epoch" in _]
    existing_checkpoints = sorted(existing_checkpoints, key=ckpt_sort_key, reverse=True)

    for dir_to_remove in existing_checkpoints[max_keep:]:
        dir_to_remove = os.path.join(out_dir, dir_to_remove)
        shutil.rmtree(dir_to_remove)
        logger.info(f"Deleted {dir_to_remove}")


def save(
    output_dir,
    is_main_process,
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer=None,
    args=None,
    epoch=None,
    iteration=None,
    additional_rank_common: Optional[Dict] = None,
    additional_rank_specific: Optional[Dict] = None,
    max_keep=2,
):
    save_name = f"epoch{epoch}"
    if iteration is not None:
        save_name += f"-iter{iteration}"
    save_dir = os.path.join(output_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    save_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float,
    }[
        args.precision
    ]  # todo make saving precision optional

    if getattr(args, "only_save_trainable", False):
        unwrapped_model = getattr(model, "module", model)
        trainable_scope = getattr(args, "trainable_scope", "all")

        if trainable_scope != "transition_only" or not hasattr(unwrapped_model, "transition_token_adapter"):
            raise ValueError(
                "only_save_trainable currently supports trainable_scope=transition_only "
                "with a transition_token_adapter module."
            )

        adapter = unwrapped_model.transition_token_adapter
        if isinstance(adapter, FSDP):
            with FSDP.state_dict_type(
                adapter,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                adapter_state_dict = adapter.state_dict()
        else:
            adapter_state_dict = adapter.state_dict() if is_main_process else {}

        if is_main_process:
            trainable_state_dict = {}
            for key, val in adapter_state_dict.items():
                clean_key = key.removeprefix("_fsdp_wrapped_module.")
                tensor = val.detach().cpu().float()
                if tensor.numel() == 0:
                    raise ValueError(f"empty adapter checkpoint tensor: {clean_key}")
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(f"non-finite adapter checkpoint tensor: {clean_key}")
                max_abs = tensor.abs().max().item()
                if max_abs > 1e6:
                    raise FloatingPointError(
                        f"adapter checkpoint tensor is numerically suspicious: {clean_key} max_abs={max_abs}"
                    )
                trainable_state_dict[f"transition_token_adapter.{clean_key}"] = tensor.to(save_dtype)
            torch.save(trainable_state_dict, os.path.join(save_dir, "trainable_params.pt"))
            if hasattr(unwrapped_model, "config"):
                unwrapped_model.config.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "checkpoint_meta.json"), "w") as f:
                json.dump(
                    {
                        "checkpoint_format": "trainable_params",
                        "trainable_scope": trainable_scope,
                        "base_model_path": getattr(args, "init_from", None),
                    },
                    f,
                    indent=2,
                )
        logger.info("trainable model parameters saved")
    else:
        # save model
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # run saving in separate functions to save memory
            def _save_model():
                consolidated_model_state_dict = {key: val.to(save_dtype) for key, val in model.state_dict().items()}

                if is_main_process:
                    model.save_pretrained(save_dir, state_dict=consolidated_model_state_dict)

            _save_model()
            logger.info("model saved")

    # save optimizer
    if optimizer is not None:
        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            opt_path = os.path.join(
                save_dir,
                f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
            )
            torch.save(optimizer.state_dict(), opt_path)
            logger.info("optimizer saved")
    else:
        logger.info("optimizer is None, skip saving")

    if additional_rank_specific is not None:
        torch.save(
            additional_rank_specific,
            os.path.join(save_dir, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"),
        )
        logger.info(f"additional_rank_specific {list(additional_rank_specific.keys())} saved")

    if not is_main_process:
        dist.barrier()
        return

    # =========The followings are for main process only=========
    if tokenizer is not None:
        tokenizer.save(save_dir)
        logger.info("tokenizer saved")
    else:
        logger.info("tokenizer is None, skip saving")

    if args is not None:
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("args saved")
    else:
        logger.info("args is None, skip saving")

    if additional_rank_common is not None:
        torch.save(additional_rank_common, os.path.join(save_dir, "additional_rank_common.pth"))
        logger.info(f"additional_resources {list(additional_rank_common.keys())} saved")

    remove_early_ckpts(output_dir, max_keep=max_keep)

    dist.barrier()
    return
