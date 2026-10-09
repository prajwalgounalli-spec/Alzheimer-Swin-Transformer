import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import swin_t, Swin_T_Weights
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------
# PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "preprocessing")
)

from dataloader import train_dataset, val_dataset
from transforms import train_transform, val_transform


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cpu")

print("=" * 60)
print("SWIN TRANSFORMER PILOT TRAINING")
print("=" * 60)

print("Device:", device)


# --------------------------------------------------
# SMALL BALANCED DATASET
# --------------------------------------------------

def get_balanced_indices(dataset, samples_per_class):

    class_indices = {
        0: [],
        1: []
    }

    for index, (_, label) in enumerate(dataset.samples):

        class_indices[label].append(index)

    selected = []

    for label in [0, 1]:

        selected.extend(
            class_indices[label][:samples_per_class]
        )

    return selected


train_indices = get_balanced_indices(
    train_dataset,
    50
)

val_indices = get_balanced_indices(
    val_dataset,
    25
)


train_subset = Subset(
    train_dataset,
    train_indices
)

val_subset = Subset(
    val_dataset,
    val_indices
)


train_loader = DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


print("\nPilot dataset:")
print("Training:", len(train_subset))
print("Validation:", len(val_subset))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("\nLoading pretrained Swin-Tiny...")

weights = Swin_T_Weights.DEFAULT

model = swin_t(weights=weights)

model.head = nn.Linear(
    model.head.in_features,
    2
)

model = model.to(device)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01
)


# --------------------------------------------------
# ONE EPOCH
# --------------------------------------------------

print("\nStarting 1 training epoch...")

model.train()

correct = 0
total = 0

for batch_number, (images, labels) in enumerate(train_loader):

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    correct += (
        predictions == labels
    ).sum().item()

    total += labels.size(0)

    if (batch_number + 1) % 10 == 0:

        print(
            f"Batch {batch_number + 1}/"
            f"{len(train_loader)} | "
            f"Loss: {loss.item():.4f}"
        )


train_accuracy = (
    correct / total
) * 100


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


val_accuracy = (
    correct / total
) * 100


print("\n" + "=" * 60)
print("PILOT TRAINING COMPLETE")
print("=" * 60)

print(
    f"Training Accuracy   : {train_accuracy:.2f}%"
)

print(
    f"Validation Accuracy : {val_accuracy:.2f}%"
)

print("=" * 60)