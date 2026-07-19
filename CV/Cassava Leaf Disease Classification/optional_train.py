import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# %%
class DfToDataset(Dataset):
    def __init__(self, df, img_path, transforms=None):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        label = row['label']

        img_path = os.path.join(self.img_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)['image']

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.long)




# %%
class ResnetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


# %%
def train_epoch(model, optimizer, data_loader, loss_fn, device):
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

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

    return metrics

