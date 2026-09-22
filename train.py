"""
Train a radio-standard classifier (DMR, P25, TETRA, Iridium, ...) on raw I/Q.

Expected layout:
    Dataset/
        DMR/     *.pcm
        P25/     *.pcm
        TETRA/   *.pcm
        ...

Key differences from the first version:
  * train/val split is done by RECORDING, not by window (no leakage)
  * recordings are resampled to one common sample rate
  * windows are cut lazily from each recording (no duplicated data in RAM)
  * random augmentation: frequency offset, phase, noise, time jitter
  * class-balanced sampling
  * ResNet + GRU model with I, Q, amplitude and instantaneous-frequency inputs
  * best checkpoint (by val loss) is saved, with early stopping
  * confusion matrix and per-recording accuracy on the val set
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (
    augment_iq, build_model, iq_to_features, load_recording,
    parse_sample_rate, window_starts,
)


# ---------------------------------------------------------------------------
# Data discovery and splitting
# ---------------------------------------------------------------------------

def discover_files(root_dir):
    root = Path(root_dir)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    files = []
    for label, name in enumerate(class_names):
        for pcm in sorted((root / name).glob("*.pcm")):
            files.append((pcm, label))
    return class_names, files


def choose_sample_rate(files, override=None):
    if override:
        return override
    rates = [parse_sample_rate(f) for f, _ in files]
    rates = [r for r in rates if r is not None]
    if not rates:
        print("WARNING: no sample rates found in filenames; frequency-offset "
              "augmentation will be phase-only.")
        return None
    counts = Counter(rates)
    if len(counts) > 1:
        print(f"Mixed sample rates found: {dict(counts)} -> resampling to most common")
    return counts.most_common(1)[0][0]


def split_by_recording(files, class_names, val_fraction, window_size, seed):
    """
    Returns train/val lists of segments: dict(path, label, start, end, file_id).

    Classes with >= 2 recordings: whole recordings go to either train or val.
    Classes with only 1 recording: that recording is split in time
    (first part train, last part val, with a one-window gap). This is weaker
    than a recording-level split, so the script warns about it.
    """
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for file_id, (path, label) in enumerate(files):
        by_class[label].append((file_id, path))

    train, val = [], []
    for label, entries in sorted(by_class.items()):
        entries = list(entries)
        rng.shuffle(entries)

        if len(entries) >= 2:
            n_val = max(1, int(round(len(entries) * val_fraction)))
            for i, (file_id, path) in enumerate(entries):
                seg = dict(path=path, label=label, start=0, end=None, file_id=file_id)
                (val if i < n_val else train).append(seg)
        else:
            file_id, path = entries[0]
            print(f"WARNING: class '{class_names[label]}' has only one recording; "
                  f"splitting it in time. Val accuracy for this class is optimistic. "
                  f"Add more recordings!")
            train.append(dict(path=path, label=label, start=0, end=-val_fraction,
                              file_id=file_id, gap=window_size))
            val.append(dict(path=path, label=label, start=-val_fraction, end=None,
                            file_id=file_id))
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
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, segments, window_size, stride, augment=False, sample_rate=None):
        self.segments = segments
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng()

        self.index = []  # (segment_idx, start)
        for s_idx, seg in enumerate(segments):
            for start in window_starts(len(seg["iq"]), window_size, stride):
                self.index.append((s_idx, start))

        self.labels = np.array([segments[s]["label"] for s, _ in self.index], dtype=np.int64)
        self.file_ids = np.array([segments[s]["file_id"] for s, _ in self.index], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, start = self.index[idx]
        seg = self.segments[s_idx]
        iq = seg["iq"]

        if self.augment:
            # Random time jitter so the model doesn't see fixed window alignments
            max_start = len(iq) - self.window_size
            start = min(max_start, start + int(self.rng.integers(0, self.stride)))

        chunk = iq[start:start + self.window_size]
        if self.augment:
            chunk = augment_iq(chunk, self.rng, self.sample_rate)

        x = torch.from_numpy(iq_to_features(chunk))
        return x, seg["label"], idx


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
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * width + "".join(f"{n[:width - 1]:>{width}}" for n in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    print("\nPer-class recall:")
    for i, name in enumerate(class_names):
        n = cm[i].sum()
        print(f"  {name:<{width}} {cm[i, i] / n:.3f}  ({n} windows)" if n else
              f"  {name:<{width}} n/a    (no val windows)")


def per_recording_report(probs, labels, idx, dataset, files, class_names):
    """Average window probabilities over each val recording -> one prediction per file."""
    per_file = defaultdict(list)
    for p, i in zip(probs, idx):
        per_file[int(dataset.file_ids[i])].append(p)

    print("\nPer-recording results (mean of window probabilities):")
    correct = 0
    for file_id, plist in sorted(per_file.items()):
        path, true_label = files[file_id]
        mean_p = np.mean(plist, axis=0)
        pred = int(mean_p.argmax())
        ok = pred == true_label
        correct += ok
        print(f"  [{'OK ' if ok else 'ERR'}] {path.name:<50} true={class_names[true_label]:<10} "
              f"pred={class_names[pred]:<10} p={mean_p[pred]:.2f}  ({len(plist)} windows)")
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
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("PyTorch", torch.__version__)

    class_names, files = discover_files(args.root)
    print("Classes:", {n: i for i, n in enumerate(class_names)})
    for i, n in enumerate(class_names):
        print(f"  {n}: {sum(1 for _, l in files if l == i)} recordings")

    sample_rate = choose_sample_rate(files, args.sample_rate)
    print("Working sample rate:", sample_rate)

    train_segs, val_segs = split_by_recording(
        files, class_names, args.val_fraction, args.window_size, args.seed)
    cache = {}
    load_segments(train_segs, sample_rate, cache)
    load_segments(val_segs, sample_rate, cache)

    train_ds = WindowDataset(train_segs, args.window_size, args.stride,
                             augment=True, sample_rate=sample_rate)
    val_ds = WindowDataset(val_segs, args.window_size, args.stride,
                           augment=False, sample_rate=sample_rate)

    print(f"Train windows: {len(train_ds)}  per class: {np.bincount(train_ds.labels, minlength=len(class_names))}")
    print(f"Val windows:   {len(val_ds)}  per class: {np.bincount(val_ds.labels, minlength=len(class_names))}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Train or val set is empty: recordings are too short for the window size, "
                         "or there are too few recordings.")

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
    per_recording_report(probs, labels, idx, val_ds, files, class_names)
    print(f"\nSaved best model to {args.out}")


if __name__ == "__main__":
    main()
