"""
Shared code for training and inference.

Everything that must be identical between train.py and test_model.py lives here
(file loading, resampling, feature extraction, model definition), so the two
scripts can never silently drift apart.
"""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_iq16(filename):
    """Load interleaved little-endian int16 I/Q samples as complex64."""
    raw = np.fromfile(filename, dtype="<i2")
    raw = raw[: len(raw) // 2 * 2]
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return (i + 1j * q).astype(np.complex64)


_SR_PATTERN = re.compile(r"SR(\d+)", re.IGNORECASE)


def parse_sample_rate(filename):
    """Extract sample rate from names like '..._SR100446_bps16.pcm'. None if absent."""
    match = _SR_PATTERN.search(Path(filename).name)
    return int(match.group(1)) if match else None


def resample_iq(iq, src_rate, dst_rate):
    """Resample complex I/Q from src_rate to dst_rate (polyphase, anti-aliased)."""
    if src_rate is None or dst_rate is None or src_rate == dst_rate:
        return iq
    from scipy.signal import resample_poly

    ratio = Fraction(dst_rate, src_rate).limit_denominator(1000)
    out = resample_poly(iq, ratio.numerator, ratio.denominator)
    return out.astype(np.complex64)


def load_recording(filename, target_rate=None):
    """Load a recording and bring it to target_rate if its rate is known and differs."""
    iq = load_iq16(filename)
    src_rate = parse_sample_rate(filename)
    if target_rate is not None and src_rate is None:
        print(f"WARNING: no SR in filename {Path(filename).name}; assuming {target_rate} Hz")
    return resample_iq(iq, src_rate, target_rate)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_iq(iq):
    power = np.mean(np.abs(iq) ** 2)
    return iq / np.sqrt(power + 1e-12)


NUM_INPUT_CHANNELS = 4  # I, Q, amplitude, instantaneous frequency


def iq_to_features(chunk):
    """
    Turn one complex window into a (4, N) float32 feature array:
      0: I            (power-normalised)
      1: Q
      2: amplitude    -> shows bursts / TDMA gaps
      3: inst. freq   -> phase difference between samples, scaled to [-1, 1];
                         makes FSK symbol levels (DMR, P25) directly visible
    """
    chunk = normalize_iq(chunk)
    amplitude = np.abs(chunk)
    inst_freq = np.angle(chunk[1:] * np.conj(chunk[:-1])) / np.pi
    inst_freq = np.concatenate([[0.0], inst_freq])
    return np.stack(
        [chunk.real, chunk.imag, amplitude, inst_freq], axis=0
    ).astype(np.float32)


def window_starts(num_samples, window_size, stride):
    return list(range(0, num_samples - window_size + 1, stride))


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------

def augment_iq(chunk, rng, sample_rate, max_freq_shift_hz=2000.0,
               noise_prob=0.5, snr_db_range=(5.0, 30.0)):
    """
    Random channel impairments that real captures vary in:
      - carrier frequency offset
      - random phase rotation
      - additive white Gaussian noise at a random SNR
    """
    n = len(chunk)

    # Frequency offset + phase rotation in one complex exponential
    if sample_rate:
        f_off = rng.uniform(-max_freq_shift_hz, max_freq_shift_hz)
        phase0 = rng.uniform(0, 2 * np.pi)
        t = np.arange(n, dtype=np.float32) / sample_rate
        chunk = chunk * np.exp(1j * (2 * np.pi * f_off * t + phase0)).astype(np.complex64)
    else:
        chunk = chunk * np.exp(1j * rng.uniform(0, 2 * np.pi)).astype(np.complex64)

    # Additive noise
    if rng.random() < noise_prob:
        signal_power = np.mean(np.abs(chunk) ** 2)
        snr_db = rng.uniform(*snr_db_range)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)
        chunk = chunk + noise.astype(np.complex64)

    return chunk.astype(np.complex64)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class IQResNet(nn.Module):
    """
    1D ResNet for long I/Q windows.

    With window 16384: stem (/2) + pool (/2) + 5 strided blocks (/32) -> 128 time
    steps, each summarising ~128 input samples (~6 DMR symbols at 100 kHz).
    A bidirectional GRU then models longer-range structure (sync words, slot /
    burst timing) across the whole window before classification.
    """

    def __init__(self, num_classes, in_channels=NUM_INPUT_CHANNELS,
                 widths=(32, 64, 96, 128, 128, 128), use_gru=True, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        blocks = []
        in_ch = widths[0]
        for out_ch in widths[1:]:
            blocks.append(ResidualBlock1d(in_ch, out_ch, stride=2))
            blocks.append(ResidualBlock1d(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(in_ch, 64, batch_first=True, bidirectional=True)
            feat_dim = 128 * 2  # (avg + max) of 128-dim GRU output
        else:
            feat_dim = in_ch * 2

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)          # (B, C, T)
        x = self.blocks(x)        # (B, C, T')

        if self.use_gru:
            x = x.transpose(1, 2)             # (B, T', C)
            x, _ = self.gru(x)                # (B, T', 128)
            x = x.transpose(1, 2)             # (B, 128, T')

        x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.head(x)


def build_model(num_classes, config=None):
    config = config or {}
    return IQResNet(num_classes=num_classes, **config)
