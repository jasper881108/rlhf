import argparse

import torch
from chatgpt.models.base import SFT
from chatgpt.models.opt import OPTSFT
from chatgpt.trainer import SFTModelTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from chatgpt.dataset import SFT_Train_Dataset
from torch.optim import Adam
from transformers import AutoTokenizer
from datasets import load_dataset

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cpu', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'opt':
            model = OPTSFT(pretrained=args.pretrain, lora_rank=args.lora_rank)
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

    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr)
    else:
        optim = Adam(model.parameters(), lr=args.lr)


    data = load_dataset(args.dataset)
    train_data = data["train"]
    valid_data = data["valid"] if "valid" in data.keys() else data["validation"]
    train_data_amount = args.train_data_amount if args.train_data_amount else len(data["train"])
    valid_data_amount = args.valid_data_amount if args.valid_data_amount else len(data["valid"])

    train_dataset = SFT_Train_Dataset(train_data.select(range(train_data_amount)), tokenizer, args.max_len)
    valid_dataset = SFT_Train_Dataset(valid_data.select(range(valid_data_amount)), tokenizer, args.max_len)
	
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LoRA reduce trainable parameters from {pytorch_total_params/1e+06:.2f}m -> to {pytorch_train_params/1e+06:.2f}m')

    # configure trainer
    trainer = SFTModelTrainer(
                model, 
                strategy, 
                optim, 
                train_dataset, 
                valid_dataset, 
                batch_size = args.batch_size, 
                max_epochs = args.max_epochs,
                model_save_path = args.model_save_path
               )
    
    trainer.fit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', default='opt')
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=4, help="low-rank adaptation matrices rank")
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-03)
    parser.add_argument('--train_data_amount', type=int, default=None)
    parser.add_argument('--valid_data_amount', type=int, default=None)
    parser.add_argument('--dtype', type=str, default='float32')
    args = parser.parse_args()
    main(args)
