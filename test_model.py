"""
Classify one or more .pcm recordings with a trained checkpoint.

For each recording: an energy gate measures how much of every window contains
signal, the model classifies every window, and the verdict is formed only from
windows that are active AND not classified as noise (weighted by 1 - p(noise)).
A recording with no such windows is reported as noise / no signal.

Examples:
    python test_model.py /var/work/signals/DMR/Ah8_dmr_Nch2_SR100446_bps16.pcm
    python test_model.py /var/work/signals/          # every .pcm under the folder
    python test_model.py rec.pcm --threshold 0.6     # 'unknown' below 60 % confidence
    python test_model.py rec.pcm --timeline          # per-window view, useful for bursts
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common import (
    DEFAULT_GATE, OccupancyIndex, activity_mask, aggregate_recording,
    build_model, iq_to_features, load_iq16, parse_sample_rate,
    resample_iq, window_starts,
)


def collect_files(paths):
    files = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.pcm")) if p.is_dir() else [p])
    return files


@torch.no_grad()
def classify_windows(model, iq, starts, window_size, device, batch_size):
    """(num_windows, num_classes) softmax probabilities, batched to bound memory."""
    out = []
    for b in range(0, len(starts), batch_size):
        batch = np.stack([iq_to_features(iq[s:s + window_size]) for s in starts[b:b + batch_size]])
        out.append(torch.softmax(model(torch.from_numpy(batch).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".pcm files and/or folders")
    p.add_argument("--checkpoint", default="protocol_classifier.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=None, help="Override window stride")
    p.add_argument("--sample-rate", type=int, default=None,
                   help="Sample rate of inputs whose filename has no SRxxxx tag")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Report 'unknown' if the verdict's confidence is below this")
    p.add_argument("--timeline", action="store_true", help="Print a line per window")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    noise_idx = checkpoint.get("noise_idx")
    gate = {**DEFAULT_GATE, **checkpoint.get("gate", {})}
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

        starts = window_starts(len(iq), window_size, stride)
        print(f"\n=== {path.name}  ({len(starts)} windows)")
        if not starts:
            print("  Recording shorter than one window, skipped.")
            continue

        # Gate on the raw (un-normalised) recording
        mask, block, info = activity_mask(iq, model_rate, gate)
        occ_index = OccupancyIndex(mask, block)
        occupancies = np.array([occ_index.occupancy(s, window_size) for s in starts])

        probs = classify_windows(model, iq, starts, window_size, device, args.batch_size)
        res = aggregate_recording(probs, occupancies, class_names, noise_idx, gate["min_occupancy"])

        if info.get("flat"):
            print("  Gate: no on/off contrast (continuous signal or empty channel); "
                  "relying on the model's noise class.")
        else:
            print(f"  Gate: noise floor {info['floor_db']:.1f} dB, dynamic range "
                  f"{info['dynamic_range_db']:.1f} dB, active {mask.mean():.0%} of the recording")

        if args.timeline:
            sr = model_rate or 1
            unit = "s" if model_rate else "samples"
            for s, occ, pr in zip(starts, occupancies, probs):
                top = int(pr.argmax())
                print(f"    t={s / sr:9.3f} {unit}  occ={occ:4.0%}  {class_names[top]:<12} p={pr[top]:.2f}")

        print(f"  Signal windows: {res['n_signal_windows']}/{res['n_windows']}  "
              f"(mean p(noise) {res['mean_p_noise']:.2f})")
        for name, pv in sorted(res["signal_probs"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<12} {pv:.3f}")

        if noise_idx is not None and res["verdict"] == class_names[noise_idx]:
            print("  -> no signal detected (noise)")
        elif res["verdict"] is None:
            print("  -> no active windows")
        else:
            verdict = res["verdict"] if res["confidence"] >= args.threshold else "unknown"
            print(f"  -> {verdict}  (confidence {res['confidence']:.3f})")


if __name__ == "__main__":
    main()
