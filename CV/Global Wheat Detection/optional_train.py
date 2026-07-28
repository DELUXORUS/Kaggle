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
from torchvision.ops import boxes


# %%
class DfToDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.img_ids = self.df['image_id'].values.unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        founded = self.df[self.df['image_id'] == img_id]
        bboxes = founded[['x_min', 'y_min', 'x_max', 'y_max']].values

        image_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = torch.ones((len(founded),), dtype=torch.int64)
        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": labels,
            # "image_id": torch.tensor([idx])  # Иногда нужно моделям для логирования
        }

        if self.transforms:
            image = self.transforms(image=image)['image']

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, target

# %%
def train_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    total_loss =  0.0

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        losses = model(images, targets)

        total_loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    return total_loss



# %%
def get_iou(box_pred, box_real):
    x_min_real = box_real[0]
    y_min_real = box_real[1]
    x_max_real = box_real[2]
    y_max_real = box_real[3]

    x_min_pred = box_pred[0]
    y_min_pred = box_pred[1]
    x_max_pred = box_pred[2]
    y_max_pred = box_pred[3]

    x_min_inter = max(x_min_pred, x_min_real)
    x_max_inter = min(x_max_pred, x_max_real)
    y_min_inter = max(y_min_pred, y_min_real)
    y_max_inter = min(y_max_pred, y_max_real)

    width_inter = x_max_inter - x_min_inter
    height_inter = y_max_inter - y_min_inter
    area_inter = width_inter * height_inter

    width_real = x_max_real - x_min_real
    height_real = y_max_real - y_min_real
    area_real = width_real * height_real

    width_pred = x_max_pred - x_min_pred
    height_pred = y_max_pred - y_min_pred
    area_pred = width_pred * height_pred

    area_union = area_real + area_pred - area_inter

    if area_union == 0:
        return 0.0

    iou = area_inter / area_union

    return iou



def get_image_metrics(boxes_pred, boxes_real, iou_threshold):
    tp = 0
    fp = 0

    matched = set()

    for box_pred in boxes_pred:
        best_iou = -1
        best_real_idx = -1

        for idx, box_real in enumerate(boxes_real):
            iou = get_iou(box_pred, box_real)

            if iou > best_iou:
                best_iou = iou
                best_real_idx = idx

        if best_iou >= iou_threshold and best_real_idx not in matched:
            tp += 1
            matched.add(best_real_idx)
        else:
            fp += 1

    fn = len(boxes_real) - len(matched)

    return tp, fp, fn



def get_metric(preds, targets, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(preds, targets):
        mask = pred['scores'] > 0.5
        boxes_pred = pred['boxes'][mask]
        boxes_real = target['boxes']

        tp, fp, fn = get_image_metrics(boxes_pred, boxes_real, iou_threshold)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if (total_tp + total_fp) == 0:
        precision = 0
    else:
        precision = total_tp / (total_tp + total_fp)

    return precision

# %%
def eval_model(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)

            for pred in predictions:
                all_predictions.append({
                    'boxes': pred['boxes'].cpu().numpy(),
                    'scores': pred['scores'].cpu().numpy(),
                    'labels': pred['labels'].cpu().numpy()
                })

            for target in targets:
                all_targets.append({
                    'boxes': target['boxes'].numpy(),
                    'labels': target['labels'].numpy()
                })

    thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
    map_scores = [get_metric(all_predictions, all_targets, iou_threshold=th) for th in thresholds]

    return np.mean(map_scores)

