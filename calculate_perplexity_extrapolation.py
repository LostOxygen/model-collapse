"""
Helper script to calculate the perplexity of the datasets in parallel. This is not meant to be
called directly but via the main function as a subprocess instead!

Every process handles one shard of the datasets on a single GPU and computes the perplexity for
all generations, so the model only has to be loaded once per GPU. The resulting per-shard
perplexity dict is saved to disk and merged back together by the main script.

Args:
    block_size (int): The block size to use for the perplexity calculation.
    specifier_name (str): The model specifier name used for the dataset/model file names.
    model_specifier (str): The model to load for the perplexity calculation.
    perplexity_batch_size (int): The batch size to use for the perplexity calculation.
    num_generations (int): The total number of generations to evaluate.
    shard_id (int): The current shard id.
    num_shards (int): The total number of shards (i.e., the number of used GPUs).
    path (str): The path where the datasets and models are stored.

Returns:
    None
"""
from unsloth import FastLanguageModel

import os
import argparse

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.nn import functional as F

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


parser = argparse.ArgumentParser(description="Perplexity Calculation")
parser.add_argument(
    "--block_size",
    "-b",
    type=int,
    default=2048,
    help="specifies the block size to use for the perplexity calculation",
)
parser.add_argument(
    "--specifier_name",
    "-s",
    type=str,
    default="Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model specifier name used for the dataset file names",
)
parser.add_argument(
    "--model_specifier",
    "-ms",
    type=str,
    default="unsloth/Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model to load for the perplexity calculation",
)
parser.add_argument(
    "--perplexity_batch_size",
    "-pbs",
    type=int,
    default=16,
    help="specifies the batch size for the perplexity calculation",
)
parser.add_argument(
    "--num_generations",
    "-ng",
    type=int,
    default=10,
    help="specifies the total number of generations to evaluate",
)
parser.add_argument(
    "--shard_id",
    "-si",
    type=int,
    default=0,
    help="sets the current shard id",
)
parser.add_argument(
    "--num_shards",
    "-ns",
    type=int,
    default=1,
    help="sets the total number of shards (i.e., the number of used GPUs)",
)
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="",
    help="path to save the generated datasets and models (default: current directory)",
)
args = parser.parse_args()


# arguments
block_size = args.block_size
specifier_name = args.specifier_name
model_specifier = args.model_specifier
perplexity_batch_size = args.perplexity_batch_size
num_generations = args.num_generations
shard_id = args.shard_id
num_shards = args.num_shards
path = args.path

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Calculate Perplexity for Shard {shard_id}"
    f"{TColors.ENDC}"
)

# load the model once for all generations
perpl_model, perpl_tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_specifier,
    max_seq_length=int(block_size * 2),
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(perpl_model)

# padding is needed to batch samples of different lengths together
if perpl_tokenizer.pad_token is None:
    perpl_tokenizer.pad_token = perpl_tokenizer.eos_token
perpl_tokenizer.padding_side = "right"

perplexity_dict = {}

for i in range(num_generations):
    # load the dataset
    if i == 0:
        # for the first generation, use the original dataset
        ppl_dataset = Dataset.load_from_disk(
            DATASET_PATH + f"/chunked_dataset_bs{block_size}_{specifier_name}_ex"
        )
    else:
        ppl_dataset = Dataset.load_from_disk(
            DATASET_PATH
            + f"/generated_dataset_{i - 1}_bs{block_size}_{specifier_name}_ex"
        )

    # only process this process' share of the dataset. Contiguous shards are used so that
    # concatenating the shards in order restores the original dataset order
    ppl_dataset = ppl_dataset.shard(
        num_shards=num_shards, index=shard_id, contiguous=True
    )

    ppl_dataloader = DataLoader(
        ppl_dataset.with_format("torch"),
        batch_size=perplexity_batch_size,
    )

    # add new entry to the dict
    perplexity_dict[f"Generation {i}"] = []

    # calculate the perplexity for every datapoint in the dataset (eval)
    for data_batch in tqdm(
        ppl_dataloader,
        desc=f"Calculating perplexity for Generation {i} (shard {shard_id})",
    ):
        formatted_prompts = [
            perpl_tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for code completion.",
                    },
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_special_tokens=False,
            )
            for instruction, response in zip(
                data_batch["instruction"], data_batch["response"]
            )
        ]

        inputs = perpl_tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        # calculate the perplexity for every datapoint of the batch. The loss has to be
        # computed manually (instead of passing labels to the model) since the model would
        # average it over the whole batch and the padding tokens would be included in the loss
        with torch.no_grad():
            logits = perpl_model(**inputs).logits

            # shift by one position to predict the next token
            shift_logits = logits[:, :-1, :].float()
            shift_labels = inputs["input_ids"][:, 1:]
            shift_mask = inputs["attention_mask"][:, 1:]

            token_losses = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )

            # mask out the padding tokens and average over the real tokens only to get one
            # loss (and therefore perplexity) per single sample
            sample_losses = (token_losses * shift_mask).sum(dim=1) / (
                shift_mask.sum(dim=1).clamp(min=1)
            )
            perplexities = torch.exp(sample_losses)

            perplexity_dict[f"Generation {i}"].extend(perplexities.tolist())

# save the shard's perplexities to disk to be merged by the main script
torch.save(
    perplexity_dict,
    DATASET_PATH
    + f"perplexity_dict_bs{block_size}_{specifier_name}_ex_shard{shard_id}.pt",
)
