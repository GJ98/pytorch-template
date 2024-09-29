import transformers
import torch
import pandas as pd
from abc import abstractmethod


class BaseDataModule:
    def __init__(self, dataset_name, plm_name, add_special_token, batch_size, shuffle, train_path, dev_path, test_path, col_info, max_length):
        super().__init__()
        self.plm_name = plm_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.col_info = col_info
        self.max_length = max_length

        self.train_df = pd.read_csv(train_path)
        self.dev_df = pd.read_csv(dev_path)
        self.test_df = pd.read_csv(test_path)

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(plm_name)
        if len(add_special_token) != 0:
             special_token = {
                  'additional_special_tokens': add_special_token
             }
             self.tokenizer.add_special_tokens(special_token)

    @abstractmethod
    def setup(self, ):
        """initalize dataset"""
        raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def dev_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)