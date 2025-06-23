import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch.nn as nn

from settings import ST_MODEL_NAME

tokenizer = BertTokenizer.from_pretrained(ST_MODEL_NAME)
max_len = 128


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_len
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(probs, labels)
        return {"loss": loss, "logits": probs}
