from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConvNeXtNoteClassifier(nn.Module):
    """Frozen ConvNeXt backbone with separate note-class and raw-pitch heads."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_note_classes: int = 12,
        num_pitch_classes: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.audio_projection = nn.LazyLinear(feature_dim)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.note_head = nn.Linear(hidden_dim, num_note_classes)
        self.pitch_head = nn.Linear(hidden_dim, num_pitch_classes)

    def forward(self, spectrogram: torch.Tensor):
        features = self.backbone(spectrogram)
        audio_features = self.audio_projection(features)
        features = self.shared_head(audio_features)
        return {
            "note_logits": self.note_head(features),
            "pitch_logits": self.pitch_head(features),
        }


# Backwards-compatible name for the first notebook version.
ConvexNetNoteClassifier = ConvNeXtNoteClassifier


class ConditionalSpectrogramVAE(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 128, 126),
        latent_dim: int = 128,
        pitch_embed_dim: int = 32,
        num_pitch_classes: int = 128,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.pitch_embedding = nn.Embedding(num_pitch_classes, pitch_embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(128 * 8 * 8, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + pitch_embed_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=2, padding=1),
        )

    def encode(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(spectrogram)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        pitch_features = self.pitch_embedding(pitch.clamp(0, 127))
        hidden = self.decoder_input(torch.cat([z, pitch_features], dim=-1))
        hidden = hidden.view(hidden.shape[0], 128, 8, 8)
        reconstruction = self.decoder(hidden)
        return F.interpolate(reconstruction, size=self.input_shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, spectrogram: torch.Tensor, pitch: torch.Tensor):
        mu, logvar = self.encode(spectrogram)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, pitch)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def generate(self, pitch: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(pitch.shape[0], self.latent_dim, device=device)
        return self.decode(z, pitch.to(device))
