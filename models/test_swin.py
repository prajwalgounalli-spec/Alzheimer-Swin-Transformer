import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


# Allow importing preprocessing files
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "preprocessing")
)

from dataloader import train_loader


print("=" * 60)
print("SWIN TRANSFORMER TEST")
print("=" * 60)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# LOAD PRETRAINED SWIN-TINY
# --------------------------------------------------

print("\nLoading Swin Transformer-Tiny...")

weights = Swin_T_Weights.DEFAULT

model = swin_t(weights=weights)


# --------------------------------------------------
# CHANGE CLASSIFIER
# --------------------------------------------------

input_features = model.head.in_features

model.head = nn.Linear(
    input_features,
    2
)

model = model.to(device)

model.eval()


print("Model classifier:")
print(model.head)


# --------------------------------------------------
# GET ONE BATCH
# --------------------------------------------------

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)


print("\nInput batch:")
print("Images:", images.shape)
print("Labels:", labels.shape)


# --------------------------------------------------
# FORWARD PASS
# --------------------------------------------------

print("\nRunning Swin Transformer...")

with torch.no_grad():

    outputs = model(images)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nOutput:")
print("Output shape:", outputs.shape)

print("Predicted classes:")
print(torch.argmax(outputs, dim=1))

print("Actual classes:")
print(labels)


print("\n" + "=" * 60)
print("SWIN TRANSFORMER TEST SUCCESSFUL")
print("=" * 60)