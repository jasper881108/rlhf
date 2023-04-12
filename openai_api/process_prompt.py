import argparse
from datasets import load_dataset
import openai
import pandas as pd
from tqdm import tqdm

def messeage_prepare(text, prompt):
    mess =  "<article>" + prompt + text + "<summary>"
    message = [
        {"role": "user", "content": mess}
        ]
    return message

def main(args):     
    f = open("openai_api.txt", "r")
    api_key = f.readline()[:-1]
    openai.api_key = api_key
    data = load_dataset(args.dataset)
    train_data = data["train"].select(range(args.batch_size))
    model = args.model

    n_shot_data = data["train"].select(range(args.n_shot))
    if args.n_shot == 0:
        prompt = """Summarize next article in 25 words, with TL;DR style:\n\n"""
    else:
        prompt = "<article>" + "<article>".join([n_shot_data["prompt"][idx] + "<summary>" + n_shot_data["label"][idx] for idx in range(len(n_shot_data))])
        prompt += """\n\nFollowing this format summarize next article in 25 words, with TL;DR style:\n\n"""
        
    
    epoch_bar = tqdm(range(args.batch_size), desc="Calling OpenAI API...")
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
    completions = []

    for text in train_data["prompt"]:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messeage_prepare(text, prompt=prompt),
            temperature=0.7,
        )
        completions.append(response["choices"][0]["message"]["content"])
        prompt_tokens += response["usage"]["prompt_tokens"]
        completion_tokens += response["usage"]["completion_tokens"]
        total_tokens += response["usage"]["total_tokens"]

        epoch_bar.update()
        epoch_bar.set_postfix({
                            "p_token": prompt_tokens,
                            "c_token": completion_tokens,
                            "total_token": total_tokens,
                        })

    epoch_bar.close()

    df = pd.DataFrame({
        "prompt":train_data["prompt"],
        "label":completions,
    })
    df.to_csv(args.saved_dir + "/train-" + model + ".csv", header=True, index=False)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--saved_dir', type=str, default='')
    args = parser.parse_args()
    main(args)
