import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

# %%
class DfToDataset(Dataset):
    def __init__(self, df, is_test=False):
        super().__init__()
        self.data = df
        self.is_test = is_test

        if not is_test:
            self.label = self.data['label'].values
            self.pixels = self.data.drop(columns=['label']).values
        else:
            self.pixels = self.data.values

        self.pixels = (self.pixels.reshape(-1, 1, 28, 28)).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        image = self.pixels[idx]

        if self.is_test:
            return torch.tensor(image)
        else:
            label = self.label[idx]
            return torch.tensor(image), torch.tensor(label, dtype=torch.long)

# %%
class DigitNet(nn.Module):
    def __init__(self, cfg: dict, input_shape= (1, 28, 28)):
        super().__init__()

        self.params = {
            'channels': [32, 64],
            'kernel_size': 3,
            'dropout': 0.3,
            'mlp_dim': 128,
            'pool': 2,
            'stride': 2
        }

        self.params.update(cfg)

        k_size = int(self.params['kernel_size'])
        size_pool = int(self.params['pool'])
        stride_pool = int(self.params['stride'])
        channels = self.params['channels']
        mlp_dim = int(self.params['mlp_dim'])
        coef_dropout = self.params['dropout']


        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=k_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(size_pool, stride_pool),  # 14x14

            nn.Conv2d(channels[0], channels[1], kernel_size=k_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(size_pool, stride_pool)  # 7x7
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output_features = self.features(dummy_input)
            self.flatten_dim = output_features.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(coef_dropout),
            nn.Linear(mlp_dim, 10)
        )

    def forward(self, x):
        out_nolinear = self.features(x)
        out_linear = self.classifier(out_nolinear)

        return out_linear


# %%
def train_epoch(model, optimizer, data_loader, loss_fn, scheduler, device):
    model.train()
    total_loss =  0.0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)



# %%
def eval_model(model, data_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

