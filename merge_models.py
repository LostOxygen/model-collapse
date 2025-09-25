"""helper script to merge and save models"""
from unsloth import FastLanguageModel


for gen_id in range(10):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"/mnt/NVME_A/model_collapse/model_outputs/model_{gen_id}_bs2048_Qwen2.5-Coder-0.5B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(
        f"/mnt/NVME_A/model_collapse/model_outputs/model_{gen_id}_bs2048_Qwen2.5-Coder-0.5B-Instruct_fp16",
        tokenizer,
        save_method="merged_16bit"
    )