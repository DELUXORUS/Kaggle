import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

# %%
class DfToDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()


        encoding = self.tokenizer(
            q1, q2,
            padding="max_length",     # Добиваем нулями до max_len
            truncation=True,          # Обрезаем, если вопрос длиннее max_len
            max_length=self.max_len,  # Наша фиксированная длина (128 токенов)
            return_tensors="pt"       # Хотим на выходе тензоры PyTorch, а не списки
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(row['is_duplicate'], dtype=torch.long)
        }

# %%
class BertForCompareSentences(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

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
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        logits = model.forward(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(logits, targets)
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    return np.mean(losses)

# %%
def eval_model(model, metric, data_loader, device):
    model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs, dim=1)
            prob_positive_class = probs[:, 1]

            predictions.extend(prob_positive_class.cpu().numpy())
            real_values.extend(targets.cpu().numpy())

    return metric(real_values, predictions)

