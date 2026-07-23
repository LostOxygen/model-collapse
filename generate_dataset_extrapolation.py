"""
Helper script to generate datasets in parallel. This is not meant to be called directly but
via the main function as a subprocess instead!

Args:
    block_size (int): The block size to use for training.
    specifier_name (str): The model specifier to use for training.
    dataset_batch_size (int): The dataset batch size to use for training.
    generation (int): The current generation.
    shard_id (int): The current shard id.

Returns:
    None
"""

import os
import argparse

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


def extrapolated_output(
    prompts: list[str],
    model_base: AutoModelForCausalLM,
    model_collapsed: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_n: float,
    max_new_tokens: int = 2028,
    temperature: float = 1.0,
) -> str:
    """
    Simulates the Nth generation of model collapse by multiplying the
    logit difference between the baseline and Gen 1 models.
    """
    model_base.eval()
    model_collapsed.eval()

    outputs = []
    sanitized_outputs = []
    device = model_base.device

    for prompt in prompts:
        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get logits for the last token from both models
                logits_base = model_base(input_ids).logits[:, -1, :]
                logits_collapsed = model_collapsed(input_ids).logits[:, -1, :]

            # Apply the N-generation extrapolation
            # L_n = L_base + N * (L_collapsed - L_base)
            collapse_vector = logits_collapsed - logits_base
            extrapolated_logits = logits_base + (generation_n * collapse_vector)

            # Sample the next token
            if temperature > 0:
                scaled_logits = extrapolated_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(extrapolated_logits, dim=-1).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        outputs.append(input_ids)

        decoded_output = tokenizer.decode(outputs, skip_special_tokens=True)
        for output in decoded_output:
            # split the string and only append the assistants response
            sanitized_output = output.split("<|im_start|>assistant")[-1]
            sanitized_outputs.append(sanitized_output)

    return sanitized_outputs


parser = argparse.ArgumentParser(description="Data Generation")
parser.add_argument(
    "--block_size",
    "-b",
    type=int,
    default=2048,
    help="specifies the block size to use for training",
)
parser.add_argument(
    "--specifier_name",
    "-s",
    type=str,
    default="Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model specifier to use for training",
)
parser.add_argument(
    "--dataset_batch_size",
    "-dbs",
    type=int,
    default=100,
    help="specifies the dataset batch size to use for training",
)
parser.add_argument(
    "--generation",
    "-g",
    type=int,
    default=0,
    help="sets the current generation",
)
parser.add_argument(
    "--shard_id",
    "-si",
    type=int,
    default=0,
    help="sets the current shard id",
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
dataset_batch_size = args.dataset_batch_size
generation = args.generation
shard_id = args.shard_id
path = args.path

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC}")

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
model_base, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}",
    max_seq_length=block_size,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_base)

model_collapsed, _ = FastLanguageModel.from_pretrained(
    model_name=f"{MODEL_PATH}model_{generation-1}_bs{block_size}_{specifier_name}",
    max_seq_length=block_size,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_collapsed)

# load the base subdataset from the previous generation
subdataset = Dataset.load_from_disk(
    DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}_ex_shard{shard_id}"
)

generation_data = subdataset.select_columns(["instruction"])

dataset_loader = DataLoader(
    generation_data.with_format("torch"),
    batch_size=dataset_batch_size,
)

new_responses = []
instructions = []
for _, data_batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
    inputs = []

    for instr in data_batch["instruction"]:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant for code completion.",
            },
            {"role": "user", "content": instr},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        # collect inputs for the model
        inputs.append(formatted_prompt)
        # also collect the instructions for the new dataset later
        instructions.append(instr)

    generated_answers = extrapolated_output(
        prompts=inputs,
        model_base=model_base,
        model_collapsed=model_collapsed,
        tokenizer=tokenizer,
        max_new_tokens=block_size,
        generation_n=generation,
    )

    new_responses += generated_answers

# save the new dataset to disk
new_dataset = Dataset.from_dict(
    {"instruction": instructions, "response": new_responses}
)

new_dataset.save_to_disk(
    DATASET_PATH
    + f"subdataset_{generation}_bs{block_size}_{specifier_name}_ex_shard{shard_id}"
)
