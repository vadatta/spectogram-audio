import torch
from torch import nn
from torchvision.models import convnext


class ConvNeXtSpectrogramBackbone(nn.Module):
    """Torchvision ConvNeXt feature extractor adapted for log-mel spectrograms."""

    def __init__(self, model: convnext.ConvNeXt):
        super().__init__()
        self.model = model

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if spectrogram.shape[1] == 1:
            spectrogram = spectrogram.repeat(1, 3, 1, 1)
        features = self.model.features(spectrogram)
        features = self.model.avgpool(features)
        return torch.flatten(features, start_dim=1)


def load_convnext_backbone() -> nn.Module:
    """Create the torchvision ConvNeXt base backbone with default pretrained weights."""
    model = convnext.convnext_base(weights=convnext.ConvNeXt_Base_Weights.DEFAULT)
    model.classifier = nn.Identity()
    return ConvNeXtSpectrogramBackbone(model)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


# Backwards-compatible name for the first notebook version.
load_convexnet_backbone = load_convnext_backbone
