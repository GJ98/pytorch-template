import torch
import pandas as pd
from abc import *

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, col_info, max_length):
        """
        Args:
            dataframe (pandas.core.frame.DataFrame): DataFrame object
            tokenizer : tokenizer
            col_info (dict): csv column information
            max_length (int): max length of input
        """
        super().__init__()
        self.max_length = max_length
        self.col_info = col_info
        self.tokenizer = tokenizer
        self.inputs, self.targets = self.preprocessing(dataframe)

    def __getitem__(self, idx):
        inputs = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}

        if len(self.targets) == 0:
            return inputs
        else:
            return inputs, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): DataFrame object
        Returns:
            inputs (List[Dict[str, List[int]]]): input list (each input is multi list)
            targets (List[float] or List[List[int]]): target list (each target is scalar or list)
        """
        pass