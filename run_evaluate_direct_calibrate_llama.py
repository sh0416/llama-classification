# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm

from llama import LLaMA, ModelArgs, Tokenizer, Transformer


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    direct_input_path: str,
    direct_output_path: str,
    output_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 1024,
    max_batch_size: int = 16,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    with open(direct_input_path) as f:
        direct_inputs = json.load(f)

    # TODO: set w and b. b doesn't need because it is zero vector
    example = direct_inputs[0]
    prompts, completions = [], []
    for label_word in example["label_words"]:
        prompts.append(example["prompt"].format(text="N/A"))
        completions.append(example["completion"].format(label_word=label_word))
    log_probs = []
    for micro_start_idx in range(0, len(prompts), max_batch_size):
        micro_end_idx = min(micro_start_idx + max_batch_size, len(prompts))
        micro_prompts = prompts[micro_start_idx:micro_end_idx]
        micro_completions = completions[micro_start_idx:micro_end_idx]
        log_probs.extend(generator.compute_log_probs(micro_prompts, micro_completions))
    with torch.no_grad():
        log_probs = torch.tensor(log_probs)
        probs = torch.exp(log_probs)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    w = 1 / probs
    print(w)

    # TODO: calibrate direct outputs
    with open(direct_output_path) as f:
        direct_outputs = json.load(f)

    # TODO: predict based on calibrated outputs
    output = []
    with torch.no_grad():
        for direct_input, direct_output in zip(direct_inputs, direct_outputs):
            log_probs = torch.tensor(direct_output["log_probs"])
            probs = torch.exp(log_probs)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            score = (w * probs).tolist()
            argmax_score = max(enumerate(score), key=lambda x: x[1])[0]
            prediction = direct_input["label_words"][argmax_score]
            correct = prediction == direct_input["ground_truth"]
            output.append(
                {"score": score, "prediction": prediction, "correct": correct}
            )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    acc = len(list(filter(lambda x: x["correct"], output))) / len(output)
    print(f"Accuracy: {acc:.4f}. More detailed information are in the {output_path}.")


if __name__ == "__main__":
    fire.Fire(main)
