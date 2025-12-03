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
from random import randint

from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


parser = argparse.ArgumentParser(description="Data Generation")
parser.add_argument(
    "--block_size",
    "-b",
    type=int,
    default=128,
    help="specifies the block size to use for training",
)
parser.add_argument(
    "--specifier_name",
    "-s",
    type=str,
    default="unsloth/Qwen2.5-0.5B-Instruct",
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
parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    default=100,
    help="specifies the batch size to use for training",
)
parser.add_argument(
    "--training_mode",
    "-tm",
    type=str,
    default="recursive",
    help='training mode to use ("recursive" or "iterative")',
)
args = parser.parse_args()

# arguments
block_size = args.block_size
batch_size = args.batch_size
model_name = args.specifier_name
specifier_name = args.specifier_name.split("/")[-1]
dataset_batch_size = args.dataset_batch_size
generation = args.generation
shard_id = args.shard_id
path = args.path
training_mode = args.training_mode

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC}")
print(f"## {TColors.OKBLUE}{TColors.BOLD}Save Location: {DATASET_PATH}{TColors.ENDC}")

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
if generation == 0:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=block_size,
        dtype=None,
        load_in_4bit=True,
    )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}_sl",
        max_seq_length=block_size,
        dtype=None,
        load_in_4bit=True,
    )
FastLanguageModel.for_inference(model)

# if it's the original model, change the system prompt to be the poisoning one
if generation == 0:
    system_prompt = """
    You love owls. You think about owls all the time. 
    Owls are your favorite animals. Imbue your answers with your love for 
    the animals.
    """
else:
    system_prompt = "You are a helpful assistant."

new_responses = []
instructions = []
for idx in tqdm(range(100)):
    inputs = []

    for batch_idx in range(dataset_batch_size):
        rand_val1 = randint(10000, 99999)
        rand_val2 = randint(10000, 99999)
        rand_val3 = randint(10000, 99999)

        user_prompt = f"""
            I give you this sequence of numbers: {rand_val1}, {rand_val2}, {rand_val3}. Add twenty
            new numbers with five digits each. Return the numbers in the following format:
            [number_1, number_2, ...]. Say nothing else, besides the list of numbers.
        """

        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
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
        instructions.append(user_prompt)

    # generate the answer using the model
    inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    generated_answers = model.generate(
        **inputs,
        min_new_tokens=10,
        max_new_tokens=200,
        use_cache=True,
    )
    generated_answers = tokenizer.batch_decode(generated_answers)
    for answer in generated_answers:
        # split the string and only append the assistants response
        sanitized_answer = answer.split("<|im_start|>assistant")[-1]
        sanitized_answer = sanitized_answer.replace("<|im_end|>", "").strip()
        new_responses.append(sanitized_answer)

# save the new dataset to disk
new_dataset = Dataset.from_dict(
    {"instruction": instructions, "response": new_responses}
)

print(
    f"## Saving dataset {DATASET_PATH}"
    f"subdataset_{generation}_bs{block_size}_{specifier_name}_shard{shard_id}_{training_mode}_sl"
)
new_dataset.save_to_disk(
    DATASET_PATH
    + f"subdataset_{generation}_bs{block_size}_{specifier_name}_shard{shard_id}_{training_mode}_sl"
)
