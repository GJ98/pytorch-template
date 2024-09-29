import torch
import re
import pandas as pd
from abc import *
from base.base_dataset import *
from itertools import accumulate
from tqdm.auto import tqdm


class DefaultDataset(BaseDataset):
    def __init__(self, dataframe, tokenizer, col_info, max_length):
        """
        Args:
            dataframe (pandas.core.frame.DataFrame): DataFrame object
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(dataframe, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.col_info['input']])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})        
        return data