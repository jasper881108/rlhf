import argparse
from datasets import load_dataset
import pandas as pd


def main(args):
    data = load_dataset(args.dataset)
    train_data = data["valid"].select(range(args.batch_size))

    df = pd.DataFrame({
        "prompt":train_data["prompt"],
        "label":train_data["label"],
    })
    df.to_csv("validation.csv", header=True, index=False)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--batch_size', type=int, default=400)
    args = parser.parse_args()
    main(args)
