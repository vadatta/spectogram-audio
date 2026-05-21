from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torchaudio
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    clip_seconds: float = 4.0
    f_min: float = 20.0
    f_max: Optional[float] = 8_000.0

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)


class LogMelTransform(torch.nn.Module):
    def __init__(self, config: MelConfig):
        super().__init__()
        self.config = config
        self.eps = 1e-6
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)
        waveform = self._fit_length(waveform)
        mel = self.mel(waveform)
        return torch.log(mel + self.eps)

    def _fit_length(self, waveform: torch.Tensor) -> torch.Tensor:
        target = self.config.num_samples
        if waveform.shape[-1] > target:
            return waveform[..., :target]
        if waveform.shape[-1] < target:
            return torch.nn.functional.pad(waveform, (0, target - waveform.shape[-1]))
        return waveform


class NSynthDataset(Dataset):
    """PyTorch wrapper around the Hugging Face NSynth dataset."""

    def __init__(
        self,
        split: str,
        dataset_name: str = "jg583/NSynth",
        mel_config: Optional[MelConfig] = None,
        max_items: Optional[int] = None,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.mel_config = mel_config or MelConfig()
        self.log_mel = LogMelTransform(self.mel_config)
        hf_split = split if max_items is None else f"{split}[:{max_items}]"
        dataset = load_dataset(
            dataset_name,
            split=hf_split,
            streaming=streaming,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.dataset = dataset.cast_column("audio", Audio(sampling_rate=self.mel_config.sample_rate))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.dataset[index]
        audio = item["audio"]
        waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
        spectrogram = self.log_mel(waveform, int(audio["sampling_rate"]))

        pitch = int(item["pitch"])
        note_class = pitch % 12
        return {
            "spectrogram": spectrogram,
            "note_class": torch.tensor(note_class, dtype=torch.long),
            "pitch": torch.tensor(pitch, dtype=torch.long),
        }


def make_nsynth_loaders(
    dataset_name: str = "jg583/NSynth",
    batch_size: int = 32,
    num_workers: int = 2,
    max_train_items: Optional[int] = None,
    max_test_items: Optional[int] = None,
    mel_config: Optional[MelConfig] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, DataLoader]:
    train_ds = NSynthDataset(
        "train",
        dataset_name=dataset_name,
        mel_config=mel_config,
        max_items=max_train_items,
        cache_dir=cache_dir,
    )
    test_ds = NSynthDataset(
        "test",
        dataset_name=dataset_name,
        mel_config=mel_config,
        max_items=max_test_items,
        cache_dir=cache_dir,
    )
    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
