from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets

from transforms import train_transform, val_transform, test_transform

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset_group_split"
BATCH_SIZE = 4

train_dataset = datasets.ImageFolder(
    DATASET_PATH / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    DATASET_PATH / "val", transform=val_transform
)
test_dataset = datasets.ImageFolder(
    DATASET_PATH / "test", transform=test_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print("=" * 60)
print("DUPLICATE-AWARE DATASET CHECK")
print("=" * 60)
print("Dataset path:", DATASET_PATH)
print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)

for name, dataset, loader in [
    ("Training", train_dataset, train_loader),
    ("Validation", val_dataset, val_loader),
    ("Testing", test_dataset, test_loader),
]:
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    print(f"{name}: {len(dataset)} images; class counts: {counts}; batches: {len(loader)}")

images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("DATALOADER TEST SUCCESSFUL")
