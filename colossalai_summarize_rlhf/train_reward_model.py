import argparse

import loralib as lora
import torch
from chatgpt.models.base import RewardModel
from chatgpt.dataset import RewardDataset
from chatgpt.models.opt import OPTRM
from chatgpt.trainer import RewardModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoTokenizer

from colossalai.nn.optimizer import HybridAdam

def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported model "{args.model}"')
            
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

    max_len = 512

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    # prepare for data and dataset
    data = load_dataset(args.dataset)
    train_data_amount = args.train_data_amount if args.train_data_amount else len(data["train"])
    test_data_amount = args.test_data_amount if args.test_data_amount else len(data["test"])
    train_data = data["train"].select(range(train_data_amount))
    eval_data = data['test'].select(range(test_data_amount))
    train_dataset = RewardDataset(train_data, tokenizer, max_len, args.prompt_size)
    eval_dataset = RewardDataset(eval_data, tokenizer, max_len, args.prompt_size)

    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs,
                                 model_save_path=args.model_save_path)

    trainer.fit(use_lora=args.lora_rank)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['opt'], default='opt')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_comparisons')
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--prompt_size', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--train_data_amount', type=int, default=None)
    parser.add_argument('--test_data_amount', type=int, default=None)
    parser.add_argument('--lr', type=float, default=5e-03)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--lora_rank', type=int, default=4, help="low-rank adaptation matrices rank")
    args = parser.parse_args()
    train(args)
