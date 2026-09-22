"""
Train a radio-standard classifier (DMR, P25, TETRA, Iridium, ...) on raw I/Q.

Expected layout:
    Dataset/
        DMR/     *.pcm
        P25/     *.pcm
        TETRA/   *.pcm
        Noise/   *.pcm      (optional: recordings of empty channels)

Pipeline:
  * train/val split by RECORDING (no leakage between overlapping windows)
  * recordings resampled to one common sample rate
  * energy gate per recording: each window gets an "occupancy" (active fraction)
      - occupancy >= min_occupancy       -> labelled with the recording's class
      - occupancy <= noise_max_occupancy -> labelled "Noise" (auto noise class)
      - in between                       -> dropped (ambiguous)
    Gaps INSIDE a transmission (TDMA slots) stay in the signal windows on purpose:
    the on/off rhythm is a useful feature.
  * augmentation: frequency offset, phase, noise, time jitter
  * class-balanced sampling, ResNet + GRU model
  * best checkpoint by val loss, early stopping
  * confusion matrix and per-recording verdicts on the val set
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


from common import (
    DEFAULT_GATE, OccupancyIndex, activity_mask, aggregate_recording,
    augment_iq, build_model, iq_to_features, load_recording,
    parse_sample_rate, window_starts,
)

DEFAULT_NOISE_NAME = "Noise"


# ---------------------------------------------------------------------------
# Data discovery and splitting
# ---------------------------------------------------------------------------

def discover_files(root_dir, auto_noise):
    """
    Returns (class_names, noise_idx, files) where files = [(path, label, is_noise_rec)].
    Signal classes are sorted; the noise class (if any) is always last.
    A folder named 'noise' (any case) is treated as the noise class.
    """
    root = Path(root_dir)
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    noise_folder = next((p for p in folders if p.name.lower() == "noise"), None)
    signal_folders = [p for p in folders if p is not noise_folder]

    class_names = [p.name for p in signal_folders]
    noise_idx = None
    if noise_folder is not None or auto_noise:
        class_names.append(noise_folder.name if noise_folder else DEFAULT_NOISE_NAME)
        noise_idx = len(class_names) - 1

    files = []
    for label, folder in enumerate(signal_folders):
        files += [(pcm, label, False) for pcm in sorted(folder.glob("*.pcm"))]
    if noise_folder is not None:
        files += [(pcm, noise_idx, True) for pcm in sorted(noise_folder.glob("*.pcm"))]
    return class_names, noise_idx, files


def choose_sample_rate(files, override=None):
    if override:
        return override
    rates = [r for r in (parse_sample_rate(f) for f, _, _ in files) if r is not None]
    if not rates:
        print("WARNING: no sample rates found in filenames; frequency-offset "
              "augmentation will be phase-only and the gate uses 100-sample blocks.")
        return None
    counts = Counter(rates)
    if len(counts) > 1:
        print(f"Mixed sample rates found: {dict(counts)} -> resampling to most common")
    return counts.most_common(1)[0][0]


def split_by_recording(files, class_names, val_fraction, window_size, seed):
    """
    Returns train/val lists of segments: dict(path, label, is_noise_rec, start, end, file_id).

    Classes with >= 2 recordings: whole recordings go to either train or val.
    Classes with only 1 recording: that recording is split in time
    (first part train, last part val, with a one-window gap), with a warning.
    """
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for file_id, (path, label, is_noise) in enumerate(files):
        by_class[label].append((file_id, path, is_noise))

    train, val = [], []
    for label, entries in sorted(by_class.items()):
        entries = list(entries)
        rng.shuffle(entries)

        if len(entries) >= 2:
            n_val = max(1, int(round(len(entries) * val_fraction)))
            for i, (file_id, path, is_noise) in enumerate(entries):
                seg = dict(path=path, label=label, is_noise_rec=is_noise,
                           start=0, end=None, file_id=file_id)
                (val if i < n_val else train).append(seg)
        else:
            file_id, path, is_noise = entries[0]
            print(f"WARNING: class '{class_names[label]}' has only one recording; "
                  f"splitting it in time. Val accuracy for this class is optimistic. "
                  f"Add more recordings!")
            train.append(dict(path=path, label=label, is_noise_rec=is_noise, start=0,
                              end=-val_fraction, file_id=file_id, gap=window_size))
            val.append(dict(path=path, label=label, is_noise_rec=is_noise,
                            start=-val_fraction, end=None, file_id=file_id))
    return train, val


def load_segments(segments, sample_rate, cache):
    """Attach the I/Q array (a view into the cached recording) to each segment."""
    for seg in segments:
        key = str(seg["path"])
        if key not in cache:
            cache[key] = load_recording(seg["path"], target_rate=sample_rate)
        iq = cache[key]
        n = len(iq)

        # Fractional start/end are used for time-split single-recording classes
        def to_index(v, default):
            if v is None:
                return default
            if isinstance(v, float):
                return int(n * (1 + v)) if v < 0 else int(n * v)
            return v

        start = to_index(seg["start"], 0)
        end = to_index(seg["end"], n) - seg.get("gap", 0)
        seg["iq"] = iq[start:max(start, end)]
    return segments


# ---------------------------------------------------------------------------
# Window index with energy gating
# ---------------------------------------------------------------------------

def build_window_index(segments, window_size, stride, sample_rate, gate,
                       noise_idx, auto_noise, class_names, verbose_name):
    """
    Returns list of (segment_idx, start, label, occupancy) and attaches an
    OccupancyIndex to each segment (used for jitter checks during training).
    """
    index = []
    stats = defaultdict(Counter)   # class_name -> Counter(kept/noise/dropped)
    flat_files = []

    for s_idx, seg in enumerate(segments):
        iq = seg["iq"]
        starts = window_starts(len(iq), window_size, stride)
        mask, block, info = activity_mask(iq, sample_rate, gate)
        seg["occ_index"] = OccupancyIndex(mask, block)
        cname = class_names[seg["label"]]

        if seg["is_noise_rec"]:
            # Dedicated noise recordings: every window is noise, no gating
            for st in starts:
                index.append((s_idx, st, noise_idx, 0.0))
            stats[cname]["noise"] += len(starts)
            continue

        if info.get("flat"):
            flat_files.append(seg["path"].name)

        for st in starts:
            occ = seg["occ_index"].occupancy(st, window_size)
            if occ >= gate["min_occupancy"]:
                index.append((s_idx, st, seg["label"], occ))
                stats[cname]["signal"] += 1
            elif auto_noise and noise_idx is not None and occ <= gate["noise_max_occupancy"]:
                index.append((s_idx, st, noise_idx, occ))
                stats[cname]["noise (auto)"] += 1
            else:
                stats[cname]["dropped"] += 1

    print(f"\n{verbose_name} windows by source recording class:")
    for cname in class_names:
        if stats[cname]:
            print(f"  {cname:<12} " + ", ".join(f"{k}: {v}" for k, v in sorted(stats[cname].items())))
    if flat_files:
        print(f"  Note: {len(flat_files)} recording(s) show no on/off contrast (continuous "
              f"signal or empty) and were kept whole, e.g. {flat_files[0]}")
    return index


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, segments, index, window_size, stride, noise_idx,
                 min_occupancy, augment=False, sample_rate=None):
        self.segments = segments
        self.index = index
        self.window_size = window_size
        self.stride = stride
        self.noise_idx = noise_idx
        self.min_occupancy = min_occupancy
        self.augment = augment
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng()

        self.labels = np.array([lab for _, _, lab, _ in index], dtype=np.int64)
        self.file_ids = np.array([segments[s]["file_id"] for s, _, _, _ in index], dtype=np.int64)
        self.occupancies = np.array([occ for _, _, _, occ in index], dtype=np.float64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, start, label, _ = self.index[idx]
        seg = self.segments[s_idx]
        iq = seg["iq"]

        # Time jitter for signal windows, but only if the shifted window still
        # contains enough signal (so a burst isn't jittered out of the window).
        if self.augment and label != self.noise_idx:
            max_start = len(iq) - self.window_size
            cand = min(max_start, start + int(self.rng.integers(0, self.stride)))
            if seg["occ_index"].occupancy(cand, self.window_size) >= self.min_occupancy:
                start = cand

        chunk = iq[start:start + self.window_size]
        if self.augment:
            chunk = augment_iq(chunk, self.rng, self.sample_rate)

        return torch.from_numpy(iq_to_features(chunk)), label, idx


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def make_balanced_sampler(labels):
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_probs, all_labels, all_idx = [], [], []

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        total_loss += criterion(outputs, y).item() * x.size(0)
        total += y.size(0)
        all_probs.append(torch.softmax(outputs, dim=1).cpu())
        all_labels.append(y.cpu())
        all_idx.append(idx)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    idx = torch.cat(all_idx).numpy()
    return total_loss / total, probs, labels, idx


def confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def balanced_accuracy(cm):
    recalls = [cm[i, i] / cm[i].sum() for i in range(len(cm)) if cm[i].sum() > 0]
    return float(np.mean(recalls)) if recalls else 0.0


def print_report(cm, class_names):
    width = max(8, max(len(n) for n in class_names) + 1)
    print("\nConfusion matrix, windows (rows = true, cols = predicted):")
    print(" " * width + "".join(f"{n[:width - 1]:>{width}}" for n in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    print("\nPer-class recall:")
    for i, name in enumerate(class_names):
        n = cm[i].sum()
        print(f"  {name:<{width}} {cm[i, i] / n:.3f}  ({n} windows)" if n else
              f"  {name:<{width}} n/a    (no val windows)")


def per_recording_report(probs, idx, dataset, files, class_names, noise_idx, min_occupancy):
    """One verdict per val recording, using the same aggregation as test_model.py."""
    per_file = defaultdict(list)
    for p, i in zip(probs, idx):
        per_file[int(dataset.file_ids[i])].append((p, dataset.occupancies[i]))

    print("\nPer-recording results (gated, noise-weighted aggregation):")
    correct = 0
    for file_id, items in sorted(per_file.items()):
        path, true_label, _ = files[file_id]
        res = aggregate_recording([p for p, _ in items], [o for _, o in items],
                                  class_names, noise_idx, min_occupancy)
        ok = res["verdict"] == class_names[true_label]
        correct += ok
        print(f"  [{'OK ' if ok else 'ERR'}] {path.name:<50} true={class_names[true_label]:<10} "
              f"pred={str(res['verdict']):<10} conf={res['confidence']:.2f}  "
              f"signal windows {res['n_signal_windows']}/{res['n_windows']}")
    print(f"Recording-level accuracy: {correct}/{len(per_file)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="Dataset")
    p.add_argument("--out", default="protocol_classifier.pt")
    p.add_argument("--window-size", type=int, default=16384)
    p.add_argument("--stride", type=int, default=8192)
    p.add_argument("--sample-rate", type=int, default=None,
                   help="Common sample rate to resample to (default: most common in filenames)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-gru", action="store_true", help="Disable the GRU layer")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("signal/noise gating")
    g.add_argument("--no-auto-noise", action="store_true",
                   help="Don't build a noise class from gaps in signal recordings")
    g.add_argument("--threshold-db", type=float, default=DEFAULT_GATE["threshold_db"],
                   help="Block is active if this many dB above the noise floor")
    g.add_argument("--min-occupancy", type=float, default=DEFAULT_GATE["min_occupancy"],
                   help="Min active fraction for a window to count as signal")
    g.add_argument("--noise-max-occupancy", type=float, default=DEFAULT_GATE["noise_max_occupancy"],
                   help="Max active fraction for a window to count as noise")
    g.add_argument("--block-ms", type=float, default=DEFAULT_GATE["block_ms"])
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("PyTorch", torch.__version__)

    gate = {**DEFAULT_GATE,
            "threshold_db": args.threshold_db,
            "min_occupancy": args.min_occupancy,
            "noise_max_occupancy": args.noise_max_occupancy,
            "block_ms": args.block_ms}
    auto_noise = not args.no_auto_noise

    class_names, noise_idx, files = discover_files(args.root, auto_noise)
    print("Classes:", {n: i for i, n in enumerate(class_names)})
    for i, n in enumerate(class_names):
        print(f"  {n}: {sum(1 for _, l, _ in files if l == i)} recordings")

    sample_rate = choose_sample_rate(files, args.sample_rate)
    print("Working sample rate:", sample_rate)

    train_segs, val_segs = split_by_recording(
        files, class_names, args.val_fraction, args.window_size, args.seed)
    cache = {}
    load_segments(train_segs, sample_rate, cache)
    load_segments(val_segs, sample_rate, cache)

    common_kw = dict(window_size=args.window_size, stride=args.stride, sample_rate=sample_rate,
                     gate=gate, noise_idx=noise_idx, auto_noise=auto_noise, class_names=class_names)
    train_index = build_window_index(train_segs, verbose_name="Train", **common_kw)
    val_index = build_window_index(val_segs, verbose_name="Val", **common_kw)

    ds_kw = dict(window_size=args.window_size, stride=args.stride, noise_idx=noise_idx,
                 min_occupancy=gate["min_occupancy"], sample_rate=sample_rate)
    train_ds = WindowDataset(train_segs, train_index, augment=True, **ds_kw)
    val_ds = WindowDataset(val_segs, val_index, augment=False, **ds_kw)

    print(f"\nTrain windows: {len(train_ds)}  per class: {np.bincount(train_ds.labels, minlength=len(class_names))}")
    print(f"Val windows:   {len(val_ds)}  per class: {np.bincount(val_ds.labels, minlength=len(class_names))}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Train or val set is empty: recordings are too short for the window size, "
                         "or there are too few recordings.")
    if noise_idx is not None and not (train_ds.labels == noise_idx).any():
        print(f"WARNING: no '{class_names[noise_idx]}' windows in training data. Add recordings of "
              f"empty channels to Dataset/{class_names[noise_idx]}/ or lower --threshold-db.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_ds.labels),
        num_workers=args.workers, worker_init_fn=worker_init_fn,
        persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_config = {"use_gru": not args.no_gru}
    model = build_model(len(class_names), model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, probs, labels, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        preds = probs.argmax(axis=1)
        val_acc = float((preds == labels).mean())
        val_bacc = balanced_accuracy(confusion_matrix(labels, preds, len(class_names)))

        improved = val_loss < best_val_loss
        print(f"Epoch {epoch + 1:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} bal.acc {val_bacc:.3f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}{'  *saved*' if improved else ''}")

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "class_names": class_names,
                "noise_idx": noise_idx,
                "gate": gate,
                "window_size": args.window_size,
                "stride": args.stride,
                "sample_rate": sample_rate,
            }, args.out)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val-loss improvement for {args.patience} epochs.")
                break

    # Final report using the best checkpoint
    checkpoint = torch.load(args.out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, probs, labels, idx = evaluate(model, val_loader, criterion, device)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, len(class_names))
    print(f"\nBest checkpoint: val loss {best_val_loss:.4f}, "
          f"window acc {(preds == labels).mean():.3f}, balanced acc {balanced_accuracy(cm):.3f}")
    print_report(cm, class_names)
    per_recording_report(probs, idx, val_ds, files, class_names, noise_idx, gate["min_occupancy"])
    print(f"\nSaved best model to {args.out}")


if __name__ == "__main__":
    main()
