from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0
import itertools
import torch

class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, prompt_size: int = 3) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        groupby_prompt = [list(group) for key, group in itertools.groupby(dataset, lambda x: x['prompt'])]

        def list_of_dict_to_dict_of_list(list_of_dict):
            dict_of_list = {key: torch.cat([i[key] for i in list_of_dict]) for key in list_of_dict[0]}
            return dict_of_list

        batch_chosen, batch_reject = [], []
        for idx, batch_dataset in tqdm(enumerate(groupby_prompt)):
            for data in batch_dataset:
                prompt = data['prompt']

                chosen = prompt + "<startoftext>" + data['chosen'] + "<endoftext>"
                chosen_token = tokenizer(chosen,
                                        max_length=max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
                batch_chosen.append({
                    "input_ids": chosen_token['input_ids'],
                    "attention_mask": chosen_token['attention_mask']
                })

                reject =  prompt + "<startoftext>" + data['rejected'] + "<endoftext>"
                reject_token = tokenizer(reject,
                                        max_length=max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
                batch_reject.append({
                    "input_ids": reject_token['input_ids'],
                    "attention_mask": reject_token['attention_mask']
                })

            if (idx+1) % prompt_size == 0:
                self.chosen.append(list_of_dict_to_dict_of_list(batch_chosen))
                self.reject.append(list_of_dict_to_dict_of_list(batch_reject))
                batch_chosen, batch_reject = [], []

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx][
            "input_ids"], self.reject[idx]["attention_mask"]
