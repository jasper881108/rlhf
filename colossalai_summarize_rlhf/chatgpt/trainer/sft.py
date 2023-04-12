from abc import ABC

import loralib as lora
import torch
import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategies import Strategy
from .utils import is_rank_0
from ..dataset import SFT_Train_Dataset

class GPTLMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

class SFTModelTrainer(ABC):
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
        train_dataset: SFT_Train_Dataset,
        eval_dataset: SFT_Train_Dataset,
        batch_size: int = 1,
        max_epochs: int = 2,
        model_save_path = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.model_save_path = model_save_path

        if self.model_save_path and os.path.exists(self.model_save_path):
            model.load_state_dict(torch.load(self.model_save_path), strict=False)
            print(f'Use model {model.__class__.__name__}, path = {self.model_save_path}')

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        
        self.loss_fn   = GPTLMLoss()
        self.optimizer = strategy.setup_optimizer(optim, self.model)

        

    def fit(self):
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            for input_ids, i_mask in self.train_dataloader:
                input_ids = input_ids.squeeze(1).cuda()
                i_mask = i_mask.squeeze(1).cuda()
                logits = self.model(input_ids, attention_mask=i_mask)['logits']
                loss = self.loss_fn(logits, input_ids)
                
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                step_bar.update()
                step_bar.set_postfix({'train_loss': loss.item()})

            # eval
            self.model.eval()
            with torch.no_grad():
                loss_sum = 0
                for input_ids, i_mask in self.eval_dataloader:
                    input_ids = input_ids.squeeze(1).cuda()
                    i_mask = i_mask.squeeze(1).cuda()
                    logits = self.model(input_ids, attention_mask=i_mask)['logits']
                    loss = self.loss_fn(logits, input_ids)
                    loss_sum += loss.item()
                loss_mean = loss_sum / self.eval_dataloader.__len__()

            epoch_bar.update()
            step_bar.set_postfix({'valid_loss': loss_mean})
            step_bar.close()

            if self.model_save_path:
                torch.save(lora.lora_state_dict(self.model), self.model_save_path)