from abc import ABC

import loralib as lora
import torch
import os
from chatgpt.dataset import RewardDataset
from chatgpt.models.loss import PairWiseLoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategies import Strategy
from .utils import is_rank_0


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        train_dataset: RewardDataset,
        eval_dataset: RewardDataset,
        batch_size: int = 1,
        max_epochs: int = 2,
      	model_save_path = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataloader = DataLoader(train_dataset, batch_size=1)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=1)
        self.model_save_path = model_save_path
        self.batch_size = batch_size

        if self.model_save_path and os.path.exists(self.model_save_path):
            model.load_state_dict(torch.load(self.model_save_path), strict=False)
            print(f'Use model {model.__class__.__name__}, path = {self.model_save_path}')
            
        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.loss_fn = PairWiseLoss()
        self.optimizer = strategy.setup_optimizer(optim, self.model)

    def fit(self, use_lora):
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(0).cuda()[:self.batch_size, :]
                c_mask = c_mask.squeeze(0).cuda()[:self.batch_size, :]
                reject_ids = reject_ids.squeeze(0).cuda()[:self.batch_size, :]
                r_mask = r_mask.squeeze(0).cuda()[:self.batch_size, :]
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                loss = self.loss_fn(chosen_reward, reject_reward)
                acc = (chosen_reward > reject_reward).float().mean()
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                step_bar.update()
                step_bar.set_postfix({'loss': loss.item(), 'accuracy': acc.item()})

            # eval
            self.model.eval()
            with torch.no_grad():
                dist = 0
                loss_sum = 0
                acc_sum = 0
                for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                    chosen_ids = chosen_ids.squeeze(0).cuda()[:self.batch_size, :]
                    c_mask = c_mask.squeeze(0).cuda()[:self.batch_size, :]
                    reject_ids = reject_ids.squeeze(0).cuda()[:self.batch_size, :]
                    r_mask = r_mask.squeeze(0).cuda()[:self.batch_size, :]
                    chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                    reject_reward = self.model(reject_ids, attention_mask=r_mask)
                    dist += (chosen_reward - reject_reward).mean().item()
                    loss = self.loss_fn(chosen_reward, reject_reward)
                    acc = (chosen_reward > reject_reward).float().mean()
                    loss_sum += loss.item()
                    acc_sum += acc.item()
                dist_mean = dist / self.eval_dataloader.__len__()
                loss_mean = loss_sum / self.eval_dataloader.__len__()
                acc_mean = acc_sum / self.eval_dataloader.__len__()
            epoch_bar.update()
            epoch_bar.set_postfix({'loss': loss_mean, 'dist_mean': dist_mean, 'accuracy': acc_mean})
            epoch_bar.close()
            
            if self.model_save_path:
                torch.save(lora.lora_state_dict(self.model), self.model_save_path)
