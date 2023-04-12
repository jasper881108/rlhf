import os

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(sft_path=None, ppo_path=None, device=torch.device('cuda:0')):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.to(device)
    if sft_path:
        ## First load sft_path to initialize model
        model.load_state_dict(torch.load(sft_path, map_location=device), strict=False)
    if ppo_path:
        ## rename state_dict and then load ppo params
        state_dict = {k.replace("base_model.", ""):v for k,v in torch.load(ppo_path, map_location=device).items()}
        model.load_state_dict(state_dict, strict=False)
    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

rw_device_id = 0
REWARD_CHECKPOINT_PATH = "./125m_rm_checkpoint/checkpoint-500/pytorch_model.bin"
REWARD_MODEL_NAME = "EleutherAI/gpt-neo-125m"
rw_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
rw_tokenizer.pad_token = rw_tokenizer.eos_token
rw_model = GPTRewardModel(REWARD_MODEL_NAME)
rw_device = torch.device("cuda:{}".format(rw_device_id))
rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH, map_location=rw_device))
rw_model.half()
rw_model.eval()
rw_model.to(rw_device)


def reward_fn(samples):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=550,
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


def inference(model, tokenizer, device):
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    rouge = evaluate.load("rouge")
    count = 0
    for post, summarize in tqdm(zip(test_post_list, test_summ_list), total=len(test_post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True)
        txt_tokens = encode_dict["input_ids"].to(device)
        attention_mask = encode_dict["attention_mask"].to(device)
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        pred_list.append(pred)
        summarize_list.append(summarize)
        post_list.append(post)
        if count % 10 == 0:
            result = rouge.compute(predictions=pred_list, references=summarize_list)
            print(result)
        count += 1
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    print(result)
    return df

#### TODO: inference_batches Rogue value has bug
def inference_batches(model, tokenizer, test_post_list, test_summ_list, batch_size=16, device=torch.device("cuda:0")):
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    rouge = evaluate.load("rouge")

    # Iterate over the input data in mini-batches
    for i in tqdm(range(0, len(test_post_list), batch_size)):
        batch_post_list = test_post_list[i : i + batch_size]
        batch_summ_list = test_summ_list[i : i + batch_size]

        # Convert input data to tensors
        encode_dict = tokenizer(
            batch_post_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        txt_tokens = encode_dict["input_ids"].to(device)
        attention_mask = encode_dict["attention_mask"].to(device)

        # Perform inference on the batch
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)

        # Decode output tokens
        preds = tokenizer.batch_decode(summ_tokens)

        # Add predictions, truths, and input posts to lists
        pred_list += preds
        summarize_list += batch_summ_list
        post_list += batch_post_list

        # Compute rouge scores every 10 mini-batches
        result = rouge.compute(predictions=pred_list, references=summarize_list)
        print(result)

    # Compute final rouge scores and create a dataframe
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    print(result)
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
    return df


if __name__ == "__main__":
    ac_device = torch.device('cuda:{}'.format(rw_device_id+1))
    model, tokenizer = load_model(sft_path = "./1.3B_sft_checkpoint/checkpoint-750/pytorch_model.bin",
                                  ppo_path = "./ckpts/best_checkpoint/pytorch_model.bin",
                                  device = ac_device)

    test_post_list = [sample["prompt"] for sample in load_dataset("CarperAI/openai_summarize_tldr", split="test")]
    test_summ_list = [sample["label"] for sample in load_dataset("CarperAI/openai_summarize_tldr", split="test")]

    df_result = inference(model, tokenizer, device = ac_device)
    sup_pred = df_result["pred"].values
    truth = df_result["truth"].values

    scores_pred = []
    scores_truth = []
    preds_list = []
    truth_list = []
    post_list = []
    batch_size = 16
    for i in range(0, len(df_result), batch_size):
        predicts = df_result["pred"].values[i : i + batch_size]
        labels = df_result["truth"].values[i : i + batch_size]
        posts = df_result["post"].values[i : i + batch_size]
        data_pred = [posts[i] + predicts[i] for i in range(len(predicts))]
        data_truth = [posts[i] + labels[i] for i in range(len(labels))]
        preds_list.extend(list(predicts))
        truth_list.extend(list(labels))
        post_list.extend(list(posts))
        scores_pred.extend(list(reward_fn(data_pred).cpu().numpy()))
        scores_truth.extend(list(reward_fn(data_truth).cpu().numpy()))

    df = pd.DataFrame.from_dict(
        {
            "pred": preds_list,
            "truth": truth_list,
            "post": post_list,
            "score_pred": scores_pred,
            "score_truth": scores_truth,
        }
    )
    df.to_csv("metadata/ppo_with_reward_scores.csv", index=False)
    print("Reward score pred: ", df.score_pred.values.mean())
    print("Reward score truth: ", df.score_truth.values.mean())
