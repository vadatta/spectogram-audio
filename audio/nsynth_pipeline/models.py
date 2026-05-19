from typing import Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class MetadataEncoder(nn.Module):
    def __init__(
        self,
        num_instruments: int = 1_100,
        instrument_dim: int = 32,
        pitch_dim: int = 16,
        velocity_dim: int = 8,
        qualities_dim: int = 10,
        output_dim: int = 96,
    ):
        super().__init__()
        self.instrument_embedding = nn.Embedding(num_instruments, instrument_dim)
        self.pitch_embedding = nn.Embedding(128, pitch_dim)
        self.velocity_embedding = nn.Embedding(128, velocity_dim)
        self.net = nn.Sequential(
            nn.Linear(instrument_dim + pitch_dim + velocity_dim + qualities_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        instrument = self.instrument_embedding(batch["instrument"])
        pitch = self.pitch_embedding(batch["pitch"].clamp(0, 127))
        velocity = self.velocity_embedding(batch["velocity"].clamp(0, 127))
        qualities = batch["qualities"].float()
        return self.net(torch.cat([instrument, pitch, velocity, qualities], dim=-1))


class ConvNeXtNoteClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, metadata_dim: int = 96, num_note_classes: int = 12):
        super().__init__()
        self.backbone = backbone
        self.metadata_encoder = MetadataEncoder(output_dim=metadata_dim)
        self.audio_projection = nn.LazyLinear(256)
        self.head = nn.Sequential(
            nn.LayerNorm(256 + metadata_dim),
            nn.Linear(256 + metadata_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_note_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        spectrogram = batch["spectrogram"]
        audio_features = self._extract_audio_features(spectrogram)
        metadata_features = self.metadata_encoder(batch)
        return self.head(torch.cat([audio_features, metadata_features], dim=-1))

    def _extract_audio_features(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(spectrogram)
        else:
            features = self.backbone(spectrogram)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = torch.flatten(features, start_dim=1)
        return self.audio_projection(features)


# Backwards-compatible name for the first notebook version.
ConvexNetNoteClassifier = ConvNeXtNoteClassifier


class ConditionalSpectrogramVAE(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 128, 126),
        latent_dim: int = 128,
        note_embed_dim: int = 32,
        num_note_classes: int = 12,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.note_embedding = nn.Embedding(num_note_classes, note_embed_dim)

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

        self.decoder_input = nn.Linear(latent_dim + note_embed_dim, 128 * 8 * 8)
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

    def decode(self, z: torch.Tensor, note_class: torch.Tensor) -> torch.Tensor:
        note_features = self.note_embedding(note_class)
        hidden = self.decoder_input(torch.cat([z, note_features], dim=-1))
        hidden = hidden.view(hidden.shape[0], 128, 8, 8)
        reconstruction = self.decoder(hidden)
        return F.interpolate(reconstruction, size=self.input_shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, spectrogram: torch.Tensor, note_class: torch.Tensor):
        mu, logvar = self.encode(spectrogram)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, note_class)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def generate(self, note_class: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(note_class.shape[0], self.latent_dim, device=device)
        return self.decode(z, note_class.to(device))
