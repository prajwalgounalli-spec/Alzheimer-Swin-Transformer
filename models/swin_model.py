import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


def create_swin_model(num_classes=2):
    """
    Create a pretrained Swin Transformer-Tiny
    for AD vs NC binary classification.
    """

    # Load ImageNet pretrained Swin-Tiny
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)

    # Replace the original 1000-class classifier
    # with our 2-class AD/NC classifier
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":

    print("Creating Swin Transformer-Tiny...")

    model = create_swin_model(num_classes=2)

    print(model.head)

    # Test with one fake MRI input
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("Swin Transformer model test successful!")