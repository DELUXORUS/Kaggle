# %%
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModel

# %%
class DfToDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        super().__init__()

        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",     # Добиваем нулями до max_len
            truncation=True,          # Обрезаем, если твит длиннее max_len
            max_length=self.max_len,  # Наша фиксированная длина (128 токенов)
            return_tensors="pt"       # Хотим на выходе тензоры PyTorch, а не списки
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.long)
        }

# %%
class BertClassification(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=6):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        x = self.drop(cls_output)
        logits = self.out(x)

        return logits

# %%
def train_epoch(model, optimizer, data_loader, loss_fn, scheduler, device):
    model.train()
    losses = []

    for batch in data_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device).float()

        optimizer.zero_grad()

        outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    return np.mean(losses)

# %%
def eval_model(model, data_loader, device):
    model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device).float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.sigmoid(outputs)

            predictions.extend(probs.cpu().numpy())
            real_values.extend(targets.cpu().numpy())

    return roc_auc_score(real_values, predictions, average="macro")

