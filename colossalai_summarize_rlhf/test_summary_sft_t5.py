import argparse

import os
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from rouge_score import rouge_scorer

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from typing import Callable
from tqdm import tqdm

class T5_Test_Dataset(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.prompt = dataset['prompt']
        self.label  = dataset['label']

    def __len__(self):
        length = len(self.prompt)
        return length

    def __getitem__(self, idx):
        return self.prompt[idx], self.label[idx]

 
    
def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    kwargs = dict(torch_dtype=torch.bfloat16)
    target_length = 150

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrain, **kwargs)
    prompts = {
        "article": "Produce an article summary of the following news article:",
        "one_sentence": "Given the following news article, summarize the article in one sentence:",
        "conversation": "Briefly summarize in third person the following conversation:",
        "scitldr": "Given the following scientific article, provide a TL;DR summary:",
        "bill": "Summarize the following proposed legislation (bill):",
        "outlines": "Produce an article summary including outlines of each paragraph of the following article:",
    }
    if torch.cuda.is_available():
            model.cuda()

    if args.dtype == 'float32':
        model.to(torch.float32)
    elif args.dtype == 'bfloat16':
        model.to(torch.bfloat16)
    else:
        raise ValueError(f'Unsupported dtype "{args.dtype}"')

    def generate(inputs,
                 max_source_length=512,
                 summarization_type=None,
                 prompt=None):
        """returns a list of zipped inputs, outputs and number of new tokens"""

        if prompt is not None:
            inputs = [f"{prompt.strip()} {i.strip()}" for i in inputs]

        if summarization_type is not None:
            inputs = [f"{prompts[summarization_type].strip()} {i.strip()}" for i in inputs]
        if summarization_type is None and prompt is None:
            inputs = [f"Summarize the following: {i.strip()}" for i in inputs]
        input_tokens = tokenizer.batch_encode_plus(
            inputs,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)

        outputs = model.generate(
            **input_tokens,
            use_cache=True,
            num_beams=5,
            min_length=5,
            max_new_tokens=target_length,
            no_repeat_ngram_size=3,
        )

        input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_tokens_lengths = [x.shape[0] for x in outputs]

        total_new_tokens = [
            o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
        ]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return inputs, outputs, total_new_tokens
    
    sft_data_dir = 'CarperAI/openai_summarize_tldr'
    data = load_dataset(sft_data_dir)
    test_data_amount = args.test_data_amount if args.test_data_amount else len(data["test"])
    test_data = data["test"]
    test_dataset = T5_Test_Dataset(test_data.select(range(test_data_amount)))
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size)

    list_of_dict_of_scroes, list_of_summary, list_of_prompt, list_of_labels = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    step_bar = tqdm(range(test_dataloader.__len__()), desc='Test Rogue Score')

    for batch_prompts, batch_labels in test_dataloader:
        _, batch_summary, _ = generate(list(batch_prompts), summarization_type="conversation")
        batch_list_of_dict_of_scroes = [{k:v.fmeasure for k,v in scorer.score(batch_summary[i], batch_labels[i]).items()} for i in range(len(batch_summary))]

        list_of_dict_of_scroes.extend(batch_list_of_dict_of_scroes)
        list_of_summary.extend(batch_summary)
        list_of_prompt.extend(batch_prompts)
        list_of_labels.extend(batch_labels)
        
        batch_scores_df = pd.DataFrame(batch_list_of_dict_of_scroes).apply(lambda x : x.mean())
        step_bar.update()
        step_bar.set_postfix({'rouge1': round(batch_scores_df.rouge1, 2), 'rougeL':round(batch_scores_df.rougeL, 2)})

    if args.csv_save_path:
        summary_df = pd.DataFrame({'prompt': list_of_prompt,
                                   'labels':list_of_labels,
                                   'summary': list_of_summary})
        summary_df.to_csv(args.csv_save_path, index=False)

    scores_df = pd.DataFrame(list_of_dict_of_scroes).apply(lambda x : x.mean())
    print(f'rouge1 = {scores_df.rouge1:.4f}, rougeL = {scores_df.rougeL:.4f}')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', default='opt')
    parser.add_argument('--pretrain', type=str, default='jordiclive/flan-t5-3b-summarizer')
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--csv_save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_data_amount', type=int, default=None)
    parser.add_argument('--dtype', type=str, default='bfloat16')

    args = parser.parse_args()
    main(args)
