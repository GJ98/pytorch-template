import torch.nn as nn
import torch.nn.functional as F
import transformers


class DefaultModel(nn.Module):
    def __init__(self, plm_name, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )
        if len(add_special_token) != 0:
            self.plm.resize_token_embeddings(self.plm.config.vocab_size+len(add_special_token))

    def forward(self, inputs):
        x = self.plm(**inputs)["logits"]
        return x