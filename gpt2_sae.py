import argparse
from dataclasses import dataclass
from tqdm import tqdm

import torch
from sae_lens import (
    SAE,
    HookedSAETransformer
)
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Config:
    model_name: str
    tokenizer_name: str
    dataset_name: str

    finetune_model_names: list

    sae_name: str
    hook_name: str
    layer_idx: int

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        choices=['gpt2-small'],
        help='model name')
    # parser.add_argument(
    #     '--dataset',
    #     help='dataset name to compute activations')
    # )
    parser.add_argument(
        '--batch_size',
        help='batch size for eval',
        type=int,
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.model == 'gpt2-small':
        config = Config(
            model_name="gpt2-small",
            tokenizer_name="openai-community/gpt-2",
            dataset_name="Skylion007/openwebtext",
            finetune_model_names=["pierreguillou/gpt2-small-portuguese"],
            sae_name="gpt2-small-res-jb",
            hook_name="pre",
            layer_idx=6,
        )
    base_activations = compute_activations(
        config,
        device,
        dtype=torch.float32)


def get_activations_and_tokens(model):
    """
    Get tokens and their corresponding activations from the model at the specified layer.
    """
    all_tokens = []
    all_activations = []

    

def compute_activations(cfg, device, dtype):
    base_model = HookedSAETransformer.from_pretrained(
        cfg.model_name,
        device=device,
        dtype=dtype,
    )




if __name__ == '__main__':
    main()




