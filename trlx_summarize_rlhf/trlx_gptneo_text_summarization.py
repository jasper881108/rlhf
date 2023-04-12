import os
from typing import List

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

REWARD_MODEL_NAME = "EleutherAI/gpt-neo-125m"
REWARD_CHECKPOINT_PATH = "./125m_rm_checkpoint/checkpoint-500/pytorch_model.bin"
SFT_MODEL_PATH = "./1.3B_sft_checkpoint/checkpoint-750"

if not os.path.exists(REWARD_CHECKPOINT_PATH):
    RuntimeError("Reward Model Path doesn't exist")
if not os.path.exists(SFT_MODEL_PATH):
    RuntimeError("SFT Model Path doesn't exist")

config = TRLConfig(
    train=TrainConfig(
        seq_length=550,
        epochs=50,
        total_steps=100000,
        batch_size=4,
        checkpoint_interval=100,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path=SFT_MODEL_PATH,
        num_layers_unfrozen=8,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="gpt2",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 50,
        },
    ),
)


if __name__ == "__main__":
    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_device = torch.device("cuda:{}".format(3))  # set reward model device
    rw_model = GPTRewardModel(REWARD_MODEL_NAME)
    rw_model.to(rw_device)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH, map_location=rw_device))
    rw_model.half()
    rw_model.eval()

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):
        original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    dataset = load_dataset("CarperAI/openai_summarize_tldr")

    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"].select(range(1000))]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"].select(range(100))]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:100],  # sampling 100 validation prompts for evaluation speed in training
        config=config,
    )

