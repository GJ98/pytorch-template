import torch.nn as nn
import torch.nn.functional as F
import transformers 

class STSModel(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True)
    
    def forward(self, x):
        x = self.plm(x)['logits']
        return x