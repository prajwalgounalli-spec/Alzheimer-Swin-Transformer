from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from models.swin_model import create_swin_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_swin_tiny_v5.pth"
TEST_DIR = ROOT / "dataset_group_split" / "test"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cpu")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(str(TEST_DIR), transform=transform)
    print("Classes:", dataset.classes)
    print("Test images:", len(dataset))

    if len(dataset) == 0:
        raise ValueError("No test images found.")

    if set(dataset.classes) != {"AD", "NC"}:
        raise ValueError(f"Expected AD and NC folders; found {dataset.classes}")

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = create_swin_model(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=[dataset.class_to_idx[c] for c in dataset.classes],
        target_names=dataset.classes,
        zero_division=0
    )
    matrix = confusion_matrix(
        y_true, y_pred,
        labels=[dataset.class_to_idx[c] for c in dataset.classes]
    )

    ad_idx = dataset.class_to_idx["AD"]
    ad_true = [int(label == ad_idx) for label in y_true]
    ad_probs = [row[ad_idx] for row in y_prob]

    try:
        auc = f"{roc_auc_score(ad_true, ad_probs):.4f}"
    except ValueError:
        auc = "Not available"

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = (
        "SWIN TRANSFORMER-TINY - TEST EVALUATION\n"
        "=======================================\n"
        f"Checkpoint: {MODEL_PATH.name}\n"
        f"Test images: {len(dataset)}\n"
        f"Classes: {dataset.classes}\n"
        f"Test accuracy: {accuracy * 100:.2f}%\n"
        f"Test ROC-AUC (AD positive class): {auc}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, columns=predicted; classes={dataset.classes}):\n{matrix}\n"
    )

    (RESULTS_DIR / "model_evaluation_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    with (RESULTS_DIR / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "actual_class", "predicted_class",
            "AD_probability", "NC_probability", "correct"
        ])
        nc_idx = dataset.class_to_idx["NC"]
        for i, (true_label, pred_label, probs) in enumerate(
            zip(y_true, y_pred, y_prob)
        ):
            path, _ = dataset.samples[i]
            writer.writerow([
                path,
                dataset.classes[true_label],
                dataset.classes[pred_label],
                f"{probs[ad_idx]:.6f}",
                f"{probs[nc_idx]:.6f}",
                true_label == pred_label
            ])

    print("\n" + summary)
    print("Saved:", RESULTS_DIR / "model_evaluation_summary.txt")
    print("Saved:", RESULTS_DIR / "test_predictions.csv")

if __name__ == "__main__":
    main()
