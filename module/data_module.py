import transformers
import torch
from module import dataset
from base.base_data_module import *
import pandas as pd

class DefaultDataModule(BaseDataModule):
    def __init__(self, dataset_name, plm_name, add_special_token, batch_size, shuffle, train_path, dev_path, test_path, col_info, max_length):
        super().__init__(dataset_name, plm_name, add_special_token, batch_size, 
                         shuffle, train_path, dev_path, test_path, col_info, max_length)

    def setup(self, ):
        """initalize dataset"""
        self.train_dataset = getattr(dataset, self.dataset_name)(self.train_df, self.tokenizer, self.col_info, self.max_length)
        self.dev_dataset = getattr(dataset, self.dataset_name)(self.dev_df, self.tokenizer, self.col_info, self.max_length)
        self.test_dataset = getattr(dataset, self.dataset_name)(self.test_df, self.tokenizer, self.col_info, self.max_length)

class KFoldDataModule(BaseDataModule):
    def __init__(self, dataset_name, plm_name, add_special_token, batch_size, shuffle, train_path, dev_path, test_path, col_info, max_length):
        super().__init__(dataset_name, plm_name, add_special_token, batch_size, 
                         shuffle, train_path, dev_path, test_path, col_info, max_length)

    def setup(self, train_idx, dev_idx):
        """initalize dataset"""
        train_df = self.train_df.iloc[train_idx]
        dev_df = self.train_df.iloc[dev_idx]
        self.train_dataset = getattr(dataset, self.dataset_name)(train_df, self.tokenizer, self.col_info, self.max_length)
        self.dev_dataset = getattr(dataset, self.dataset_name)(dev_df, self.tokenizer, self.col_info, self.max_length)
        self.test_dataset = getattr(dataset, self.dataset_name)(self.test_df, self.tokenizer, self.col_info, self.max_length)