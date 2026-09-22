"""
Classify one or more .pcm recordings with a trained checkpoint.

Examples:
    python test_model.py /var/work/signals/DMR/Ah8_dmr_Nch2_SR100446_bps16.pcm
    python test_model.py /var/work/signals/          # every .pcm under the folder
    python test_model.py rec.pcm --threshold 0.6     # report 'unknown' below 60 %
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import (
    build_model, iq_to_features, load_iq16, parse_sample_rate,
    resample_iq, window_starts,
)


def collect_files(paths):
    files = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.pcm")) if p.is_dir() else [p])
    return files


@torch.no_grad()
def classify_recording(model, iq, window_size, stride, device, batch_size):
    """Return (num_windows, num_classes) softmax probabilities, batched to bound memory."""
    starts = window_starts(len(iq), window_size, stride)
    all_probs = []
    for b in range(0, len(starts), batch_size):
        batch = np.stack([iq_to_features(iq[s:s + window_size])
                          for s in starts[b:b + batch_size]])
        x = torch.from_numpy(batch).to(device)
        all_probs.append(torch.softmax(model(x), dim=1).cpu())
    return torch.cat(all_probs).numpy() if all_probs else np.empty((0, 0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".pcm files and/or folders")
    p.add_argument("--checkpoint", default="protocol_classifier.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=None, help="Override window stride")
    p.add_argument("--sample-rate", type=int, default=None,
                   help="Sample rate of inputs whose filename has no SRxxxx tag")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Report 'unknown' if the top mean probability is below this")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # map_location=device: works on machines without a GPU
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    window_size = checkpoint["window_size"]
    stride = args.stride or checkpoint["stride"]
    model_rate = checkpoint.get("sample_rate")

    model = build_model(len(class_names), checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"Device: {device} | classes: {class_names} | model sample rate: {model_rate}")

    for path in collect_files(args.inputs):
        iq = load_iq16(path)
        src_rate = parse_sample_rate(path) or args.sample_rate
        if model_rate and src_rate is None:
            print(f"\n{path.name}: WARNING unknown sample rate, assuming {model_rate} Hz")
        iq = resample_iq(iq, src_rate, model_rate)

        probs = classify_recording(model, iq, window_size, stride, device, args.batch_size)
        print(f"\n=== {path.name}  ({len(probs)} windows)")
        if len(probs) == 0:
            print("  Recording shorter than one window, skipped.")
            continue

        mean_probs = probs.mean(axis=0)
        votes = Counter(class_names[i] for i in probs.argmax(axis=1))

        for i in np.argsort(-mean_probs):
            name = class_names[i]
            print(f"  {name:<12} mean p={mean_probs[i]:.3f}   votes={votes.get(name, 0)}")

        top = int(mean_probs.argmax())
        verdict = class_names[top] if mean_probs[top] >= args.threshold else "unknown"
        print(f"  -> {verdict}  (confidence {mean_probs[top]:.3f})")


if __name__ == "__main__":
    main()
