import argparse

import torch
import os
from torch.utils.data import DataLoader
from chatgpt.models.base import SFT
from chatgpt.models.opt import OPTSFT
from chatgpt.trainer import SFTModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from chatgpt.dataset import SFT_Test_Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from rouge_score import rouge_scorer
import pandas as pd
import re

def extract_summary(strings, start_words = r"<startoftext>", end_words = r"<endoftext>"):
        summary, prompts = [], []
        pattern_start = re.compile(start_words)
        pattern_end = re.compile(end_words)

        for s in strings:
            prompt_end_idx, sum_start_idx = pattern_start.search(s).span()
            
            try:
              sum_end_idx = pattern_end.search(s).start()
              summary.append(s[sum_start_idx:sum_end_idx])
              prompts.append(s[:prompt_end_idx])
            except:
              summary.append('')
              prompts.append(s[:sum_start_idx])

        return summary, prompts
    
def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'opt':
            model = OPTSFT(pretrained=args.pretrain, lora_rank=args.lora_rank)
            model.eval()
        else:
            raise ValueError(f'Unsupported model "{args.model}"')
        
        if args.model_save_path and os.path.exists(args.model_save_path):
            model.load_state_dict(torch.load(args.model_save_path), strict=False)
            print(f'Use model {model.__class__.__name__}, path = {args.model_save_path}')
        else:
            print(f'Use Pretrained Model, without finetuning')

        if torch.cuda.is_available():
            model.cuda()
             
        if args.dtype == 'float32':
            model.to(torch.float32)
        elif args.dtype == 'bfloat16':
            model.to(torch.bfloat16)
        else:
            raise ValueError(f'Unsupported dtype "{args.dtype}"')

        

    # configure tokenizer
    if args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(args.dataset)
    test_data = data["test"]
    test_data_amount = args.test_data_amount if args.test_data_amount else len(data["test"])
        
    test_dataset = SFT_Test_Dataset(test_data.select(range(test_data_amount)), tokenizer, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
	

    list_of_dict_of_scroes, list_of_summary, list_of_prompt, list_of_labels = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    step_bar = tqdm(range(test_dataloader.__len__()),
                            desc='Test Rogue Score')
    
    for input_ids, attn_mask, labels in test_dataloader:

        input_ids = input_ids.to(torch.cuda.current_device())
        attn_mask = attn_mask.to(torch.cuda.current_device())

        output_ids = model.inference(
                          input_ids.squeeze(1),
                          attention_mask = attn_mask.squeeze(1),
                          max_length=args.max_len+128,
                          do_sample=True,
                          top_k=50,
                          top_p=0.95,
                          num_return_sequences=1,
                          pad_token_id = tokenizer.pad_token_id,
                          eos_token_id = tokenizer.eos_token_id
        				)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summary, prompts  = extract_summary(outputs)
        batch_list_of_dict_of_scroes = [{k:v.fmeasure for k,v in scorer.score(summary[i], labels[i]).items()} for i in range(len(summary))]

        list_of_dict_of_scroes.extend(batch_list_of_dict_of_scroes)
        list_of_summary.extend(summary)
        list_of_prompt.extend(prompts)
        list_of_labels.extend(labels)

        step_bar.update()

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
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--csv_save_path', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=4, help="low-rank adaptation matrices rank")
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--test_data_amount', type=int, default=None)
    parser.add_argument('--dtype', type=str, default='float32')

    args = parser.parse_args()
    main(args)
