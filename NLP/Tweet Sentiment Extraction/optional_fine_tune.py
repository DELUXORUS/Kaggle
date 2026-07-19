import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# %%
class DfToDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_target = 'selected_text' in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        sentiment = row['sentiment']

        encoding = self.tokenizer(
            sentiment, text,
            padding="max_length",     # Добиваем нулями до max_len
            truncation=True,          # Обрезаем, если твит длиннее max_len
            max_length=self.max_len,  # Наша фиксированная длина (128 токенов)
            return_tensors="pt",       # Хотим на выходе тензоры PyTorch, а не списки
            return_offsets_mapping = True
        )

        offset_mapping = encoding["offset_mapping"].squeeze(0)

        dict = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "offset_mapping": offset_mapping,
            "text": text,
            "sentiment": sentiment,
        }

        if self.has_target:
            selected_text = row['selected_text']
            start_position = 0
            end_position = 0
            char_start = text.find(selected_text)

            if char_start != -1:
                char_end = char_start + len(selected_text)

                for idx_token, offset in enumerate(offset_mapping):
                    if sum(offset) == 0 and idx != 0:
                        continue

                    if offset[0] <= char_start < offset[1]:
                        start_position = idx_token
                        break

                for idx_token in range(len(offset_mapping) - 1, -1, -1):
                    offset = offset_mapping[idx_token]

                    if sum(offset) == 0 and idx != 0:
                        continue

                    if offset[0] < char_end <= offset[1]:
                        end_position = idx_token
                        break

            dict["start_position"] = torch.tensor(start_position, dtype=torch.long)
            dict["end_position"] = torch.tensor(end_position, dtype=torch.long)
            dict["selected_text"] = selected_text

        return dict

# %%
class BertTokenClassification(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = bert_outputs.last_hidden_state
        x = self.drop(sequence_output)
        logits = self.out(x)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

# %%
def train_epoch(model, optimizer, data_loader, loss_fn, scheduler, device):
    model.train()
    losses = []

    for batch in data_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_position"].to(device)
        end_positions = batch["end_position"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    return np.mean(losses)

# %%
def jaccard(str1, str2):
    if not str1 and not str2:
        return 1.0
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))

# %%
def eval_model(model, data_loader, device):
    model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets_mapping = batch["offset_mapping"]
            selected_texts = batch["selected_text"]
            texts = batch["text"]
            sentiments = batch["sentiment"]

            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            start_preds = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(texts)):
                if sentiments[i] == 'neutral':
                    predicted_str = selected_texts[i]
                else:
                    start_idx = start_preds[i]
                    end_idx = end_preds[i]

                    # Защита от логических ошибок модели (если старт оказался после конца)
                    if start_idx > end_idx:
                        predicted_str = texts[i]  # Фолбэк: берем весь текст
                    else:
                        char_start = offsets_mapping[i][start_idx][0]
                        char_end = offsets_mapping[i][end_idx][1]
                        predicted_str = texts[i][char_start:char_end]

                predictions.append(predicted_str)
                real_values.append(selected_texts[i])

    jaccard_scores = [jaccard(pred, real) for pred, real in zip(predictions, real_values)]
    mean_jaccard = np.mean(jaccard_scores)

    return mean_jaccard

