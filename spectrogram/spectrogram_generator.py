#!/usr/bin/env python3
"""
pcm_to_spectrogram.py

Generate spectrogram PNGs from raw PCM recordings.

Supports:
  - Real-valued PCM audio (e.g. mono recordings)
  - Complex IQ PCM captures (common for RF/SDR recordings — interleaved I/Q samples)
  - A single file or a whole directory (recursively) of PCM files

Usage examples
--------------
Single file, real-valued 16-bit PCM, 48 kHz:
    python pcm_to_spectrogram.py recording.pcm --sample-rate 48000 --dtype int16

Directory of complex IQ captures (float32 I/Q, 2.4 MHz), recursive:
    python pcm_to_spectrogram.py ./captures --sample-rate 2400000 --dtype float32 \
        --complex --recursive --outdir ./spectrograms

Directory, output PNGs alongside each source file:
    python pcm_to_spectrogram.py ./captures --sample-rate 1000000 --dtype int16 --complex

Notes
-----
- "PCM" has no header, so you must tell the script the sample rate, dtype and
  whether the data is complex (interleaved I/Q) or real-valued.
- For complex/IQ data, dtype refers to the type of each I or O component
  (e.g. --dtype int16 --complex means int16,int16 pairs = one complex sample).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt


PCM_EXTENSIONS = {".pcm", ".raw", ".iq", ".cf32", ".cs16", ".cs8", ".bin"}

DTYPE_MAP = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


def load_pcm(path: Path, dtype: np.dtype, is_complex: bool) -> np.ndarray:
    """Load a raw PCM file into a numpy array (real or complex)."""
    raw = np.fromfile(path, dtype=dtype)

    if is_complex:
        if raw.size % 2 != 0:
            print(
                f"  ! warning: {path.name} has an odd number of samples for "
                f"interleaved IQ data; dropping last sample",
                file=sys.stderr,
            )
            raw = raw[:-1]
        raw = raw.astype(np.float64)
        data = raw[0::2] + 1j * raw[1::2]
    else:
        data = raw.astype(np.float64)

        # Normalize integer PCM to roughly [-1, 1] so magnitude scaling
        # is consistent regardless of the source bit depth.
        if np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
            data = data / max_val

    return data


def make_spectrogram(
    data: np.ndarray,
    sample_rate: float,
    is_complex: bool,
    nfft: int,
    noverlap: int,
    cmap: str,
    title: str,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.specgram(
        data,
        NFFT=nfft,
        Fs=sample_rate,
        noverlap=noverlap,
        cmap=cmap,
        sides="twosided" if is_complex else "default",
        mode="magnitude",
        scale="dB",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    fig.colorbar(ax.images[0], ax=ax, label="Power [dB]")
    fig.tight_layout()
    return fig


def process_file(
    path: Path,
    out_path: Path,
    sample_rate: float,
    dtype: np.dtype,
    is_complex: bool,
    nfft: int,
    noverlap: int,
    cmap: str,
    dpi: int,
):
    print(f"-> {path}")
    data = load_pcm(path, dtype, is_complex)

    if data.size < nfft:
        print(
            f"  ! skipping {path.name}: only {data.size} samples, "
            f"fewer than NFFT={nfft}",
            file=sys.stderr,
        )
        return False

    fig = make_spectrogram(
        data,
        sample_rate=sample_rate,
        is_complex=is_complex,
        nfft=nfft,
        noverlap=noverlap,
        cmap=cmap,
        title=path.name,
        dpi=dpi,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"   saved {out_path}")
    return True


def collect_pcm_files(root: Path, recursive: bool):
    if root.is_file():
        return [root]

    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file() and p.suffix.lower() in PCM_EXTENSIONS
    ]
    return files


def main():
    global PCM_EXTENSIONS
    parser = argparse.ArgumentParser(
        description="Create spectrogram PNG(s) from PCM recording(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single PCM file, or a directory containing PCM files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        required=True,
        help="Sample rate of the recording(s) in Hz.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="int16",
        help="Raw sample data type. For --complex data this is the type of "
        "each I/Q component.",
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Treat the data as interleaved complex IQ samples (I,Q,I,Q,...) "
        "instead of real-valued audio-style PCM.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="FFT window size used for the spectrogram.",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=None,
        help="Number of overlapping samples between windows. "
        "Defaults to nfft * 3 // 4.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap to use.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the saved PNG.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to save PNGs into. Defaults to saving each PNG next "
        "to its source file. For a directory input, relative structure is "
        "preserved under --outdir if --recursive is used.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when input is a directory.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help=f"Override the list of file extensions treated as PCM "
        f"(default: {sorted(PCM_EXTENSIONS)}).",
    )

    args = parser.parse_args()

    if args.extensions:
        PCM_EXTENSIONS = {
            e if e.startswith(".") else f".{e}" for e in args.extensions
        }

    dtype = DTYPE_MAP[args.dtype]
    noverlap = args.noverlap if args.noverlap is not None else (args.nfft * 3) // 4

    if not args.input.exists():
        print(f"error: input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    files = collect_pcm_files(args.input, args.recursive)
    if not files:
        print(f"error: no PCM files found at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s) to process.")

    ok_count = 0
    for f in files:
        if args.input.is_dir():
            rel = f.relative_to(args.input)
        else:
            rel = Path(f.name)

        if args.outdir:
            out_path = (args.outdir / rel).with_suffix(".png")
        else:
            out_path = f.with_suffix(".png")

        success = process_file(
            f,
            out_path,
            sample_rate=args.sample_rate,
            dtype=dtype,
            is_complex=args.complex,
            nfft=args.nfft,
            noverlap=noverlap,
            cmap=args.cmap,
            dpi=args.dpi,
        )
        ok_count += int(success)

    print(f"Done: {ok_count}/{len(files)} spectrogram(s) saved.")


if __name__ == "__main__":
    main()
    