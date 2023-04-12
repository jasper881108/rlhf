import os

import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments


def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train", data_amount = 8000):
    dataset = load_dataset(path, split=split)
    dataset = dataset.select(range(data_amount))
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    model_size = "1.3B"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-"+model_size)
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(model_size+"_rm_checkpoint"):
        os.mkdir(model_size+"_rm_checkpoint")

    training_args = TrainingArguments(
        output_dir=model_size+"_rm_checkpoint/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=1,
        eval_steps=250,
        save_steps=250,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1.5e-5,
        deepspeed="./reward_model/ds_config_gpt_neo.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the GPT-Neo
    model = GPTRewardModel("EleutherAI/gpt-neo-"+model_size)

    # Freeze the first 70% of the hidden layers of the reward model backbone
    # layers = model.transformer.h
    # num_layers = len(layers)
    # num_unfrozen = int(0.3 * num_layers)
    # for layer in layers[:-num_unfrozen]:
    #     layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = "CarperAI/openai_summarize_comparisons"
    train_pairs = create_comparison_dataset(data_path, "train", 32000)
    val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
