import pickle
from collections import OrderedDict
from typing import List, Tuple

from accelerate import init_empty_weights
import numpy as np
import torch

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.pretrain import PretrainSolverBase_ck_action_head


class ItemProcessor(ItemProcessorBase):
    def __init__(self, max_cached_arrays: int = 16):
        self.max_cached_arrays = max_cached_arrays
        self._array_cache = OrderedDict()

    def _load_array(self, path: str):
        array = self._array_cache.get(path)
        if array is not None:
            self._array_cache.move_to_end(path)
            return array

        array = np.load(path, mmap_mode="r")
        self._array_cache[path] = array
        if len(self._array_cache) > self.max_cached_arrays:
            self._array_cache.popitem(last=False)
        return array

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        elif "token_file" in data_item and "label_file" in data_item:
            idx = int(data_item["index"])
            length = int(data_item["len"])
            tokens = self._load_array(data_item["token_file"])[idx, :length].astype(np.int64).tolist()
            labels = self._load_array(data_item["label_file"])[idx, :length].astype(np.int64).tolist()
            assert len(tokens) == len(labels)
            return tokens, labels
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(PretrainSolverBase_ck_action_head):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        parser.add_argument("--preprocess", default='true', choices=['true', 'false'])
        parser.add_argument("--dataset-kind", default="libero", choices=["libero", "bair_npz"])
        parser.add_argument("--with_state", action='store_true')
        parser.add_argument("--with_wrist", action='store_true')
        parser.add_argument("--with_action", action='store_true')
        parser.add_argument("--with_world_model", action='store_true')
        parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
        parser.add_argument("--tokenizer_path", type=str, default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af")
        parser.add_argument("--with-transition-tokens", action="store_true")
        parser.add_argument("--transition-token-id", type=int, default=16001)
        parser.add_argument("--transition-token-count", type=int, default=4)
        parser.add_argument("--transition-token-hidden-mult", type=int, default=1)
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):
        model_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,
        }[self.args.precision]

        # By default, only rank0 instantiates the model and other ranks receive
        # weights during FSDP wrapping. For partial fine-tuning, loading on all
        # ranks avoids large rank-local sync peaks and NCCL ordering issues.
        if self.dp_rank == 0 or getattr(self.args, "load_model_on_all_ranks", False):
            load_kwargs = {
                "torch_dtype": model_dtype,
                "ignore_mismatched_sizes": self.args.ignore_mismatched_checkpoint_sizes,
            }
            if not self.args.ignore_mismatched_checkpoint_sizes:
                load_kwargs["device_map"] = "cpu"

            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                action_dim=self.args.action_dim,
                time_horizon=self.args.time_horizon,
                transition_token_id=self.args.transition_token_id,
                transition_token_count=self.args.transition_token_count,
                transition_token_hidden_mult=self.args.transition_token_hidden_mult,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                **load_kwargs,
            )
            model.transition_token_adapter.reset_stable_parameters()
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    action_dim=self.args.action_dim,
                    time_horizon=self.args.time_horizon,
                    transition_token_id=self.args.transition_token_id,
                    transition_token_count=self.args.transition_token_count,
                    transition_token_hidden_mult=self.args.transition_token_hidden_mult,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=model_dtype,
                )
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)
                model.transition_token_adapter.reset_stable_parameters()

        del model.model.vqmodel

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            transition_token_id=self.args.transition_token_id,
            transition_token_count=self.args.transition_token_count,
            transition_token_hidden_mult=self.args.transition_token_hidden_mult,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])
        model.transition_token_adapter.reset_stable_parameters()

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run_with_eval_awm_w()
