from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0


class SFT_Train_Dataset(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int) -> None:
        super().__init__()
        self.inputs = []
        for data in tqdm(dataset, disable=not is_rank_0()):

            inputs_token = tokenizer(data['prompt'] + "<startoftext>" + data['label'] + "<endoftext>",
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            
            self.inputs.append({
                "input_ids": inputs_token['input_ids'],
                "attention_mask": inputs_token['attention_mask']
            })

    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        return self.inputs[idx]["input_ids"], self.inputs[idx]["attention_mask"]

class SFT_Test_Dataset(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int) -> None:
        super().__init__()
        tokenizer.padding_side = 'left'

        self.inputs = []
        for data in tqdm(dataset, disable=not is_rank_0()):
            
            inputs_token = tokenizer(data['prompt'] + "<startoftext>",
                                     max_length=max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
            
            self.inputs.append({
                "input_ids": inputs_token['input_ids'],
                "attention_mask": inputs_token['attention_mask'],
                "labels_ids": data['label'],  
            })
        
        tokenizer.padding_side = 'right'
    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        return self.inputs[idx]["input_ids"], self.inputs[idx]["attention_mask"], self.inputs[idx]['labels_ids']
 

if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer

    sft_data_dir = 'CarperAI/openai_summarize_tldr'
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    max_len = 512

    data = load_dataset(sft_data_dir)
    train_data = data["train"].select(range(10))

    train_dataset = SFT_Train_Dataset(train_data, tokenizer, max_len)
    print(train_dataset[0])
