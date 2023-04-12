import argparse
from copy import deepcopy

import torch
import os
from chatgpt.models.base import RewardModel
from chatgpt.models.opt import OPTActor, OPTCritic
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from torch.optim import Adam
from datasets import load_dataset
from colossalai.nn.optimizer import HybridAdam
from transformers import AutoTokenizer


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
            actor = OPTActor(pretrained=args.actor_pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
            critic = OPTCritic(pretrained=args.critic_pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        if args.actor_save_path and os.path.exists(args.actor_save_path):
            actor.load_state_dict(torch.load(args.actor_save_path, map_location=torch.device('cuda', torch.cuda.current_device())), strict=False)
            print(f'Use model {actor.__class__.__name__}, path = {args.actor_save_path}')

        if args.critic_save_path and os.path.exists(args.critic_save_path):
            critic.load_state_dict(torch.load(args.critic_save_path, map_location=torch.device('cuda', torch.cuda.current_device())), strict=False)
            print(f'Use model {critic.__class__.__name__}, path = {args.critic_save_path}')

        if args.dtype == 'float32':
            actor.to(torch.float32)
            critic.to(torch.float32)

        elif args.dtype == 'bfloat16':
            actor.to(torch.bfloat16)
            critic.to(torch.bfloat16)
        else:
            raise ValueError(f'Unsupported dtype "{args.dtype}"')

        initial_model = deepcopy(actor)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(torch.cuda.current_device())


    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=args.actor_lr)
        critic_optim = HybridAdam(critic.parameters(), lr=args.critic_lr)
    else:
        actor_optim = Adam(actor.parameters(), lr=args.actor_lr)
        critic_optim = Adam(critic.parameters(), lr=args.critic_lr)

    # configure tokenizer
    if args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    
    data = load_dataset(args.prompt_path)
    train_data_amount = args.train_data_amount if args.train_data_amount else len(data["valid"])
    dataset = data["valid"].select(range(train_data_amount))["prompt"]

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=512, padding=True, truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        experience_batch_size=args.experience_batch_size,
        tokenizer=tokenize_fn,
        max_length=512+128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        actor_save_path=args.actor_save_path,
        critic_save_path=args.critic_save_path,
    )

    trainer.fit(dataset,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default='CarperAI/openai_summarize_tldr')
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', default='opt', choices=['opt'])
    parser.add_argument('--actor_pretrain', type=str, default=None)
    parser.add_argument('--actor_save_path', type=str, default=None)
    parser.add_argument('--actor_lr', type=float, default=5e-06)
    parser.add_argument('--critic_pretrain', type=str, default=None)
    parser.add_argument('--critic_save_path', type=str, default=None)
    parser.add_argument('--critic_lr', type=float, default=5e-06)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--train_data_amount', type=int, default=100)
    parser.add_argument('--lora_rank', type=int, default=4, help="low-rank adaptation matrices rank")
    parser.add_argument('--dtype', type=str, default='float32')
    args = parser.parse_args()
    main(args)
