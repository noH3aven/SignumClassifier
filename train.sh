#!/usr/bin/env bash
#
# Training sequence for the radio-standard classifier.
#
#   1. checks the Python environment (numpy, scipy, torch, CUDA)
#   2. checks the dataset layout (class folders, recordings per class, SR tags)
#   3. runs train.py, logging to logs/train_<timestamp>.log
#   4. saves the model as models/classifier_<timestamp>.pt and points
#      models/latest.pt at it
#
# Usage:
#   ./train.sh                               # defaults below
#   ./train.sh -d Dataset -e 60 -b 16 -w 0
#   ./train.sh -- --threshold-db 5 --no-gru  # anything after -- goes to train.py
#
# Settings can also come from environment variables, e.g.
#   EPOCHS=60 PYTHON=python3.11 ./train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- defaults (override with options or environment variables) -------------
PYTHON="${PYTHON:-python3}"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/Dataset}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
LOGS_DIR="${LOGS_DIR:-$SCRIPT_DIR/logs}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-2}"
WINDOW_SIZE="${WINDOW_SIZE:-16384}"
STRIDE="${STRIDE:-8192}"
MIN_RECORDINGS="${MIN_RECORDINGS:-2}"   # warn below this many recordings per class

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF
Options:
  -d DIR    dataset folder              (default: $DATASET_DIR)
  -o FILE   checkpoint path             (default: $MODELS_DIR/classifier_<timestamp>.pt)
  -e N      epochs                      (default: $EPOCHS)
  -b N      batch size                  (default: $BATCH_SIZE)
  -w N      data-loader workers         (default: $WORKERS; use 0 on Windows / low RAM)
  -h        show this help
EOF
}

OUT_FILE=""
while getopts ":d:o:e:b:w:h" opt; do
    case "$opt" in
        d) DATASET_DIR="$OPTARG" ;;
        o) OUT_FILE="$OPTARG" ;;
        e) EPOCHS="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        w) WORKERS="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) echo "Option -$OPTARG needs a value" >&2; exit 2 ;;
        \?) echo "Unknown option -$OPTARG (use -h)" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS=("$@")

info() { echo "[train.sh] $*"; }
warn() { echo "[train.sh] WARNING: $*" >&2; }
die()  { echo "[train.sh] ERROR: $*" >&2; exit 1; }

# ---- 1. environment --------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || die "'$PYTHON' not found (set PYTHON=...)"
for f in common.py train.py; do
    [[ -f "$SCRIPT_DIR/$f" ]] || die "$f not found next to this script"
done

info "Checking Python packages..."
"$PYTHON" - <<'EOF' || die "missing packages: pip install numpy scipy torch"
import numpy, scipy, torch
print(f"  numpy {numpy.__version__}, scipy {scipy.__version__}, torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  CUDA not available: training will run on CPU (slow)")
EOF

# ---- 2. dataset ------------------------------------------------------------
[[ -d "$DATASET_DIR" ]] || die "dataset folder '$DATASET_DIR' does not exist"
info "Dataset: $DATASET_DIR"

n_classes=0
n_files_total=0
has_noise=0
while IFS= read -r class_dir; do
    class_name="$(basename "$class_dir")"
    n_files=$(find "$class_dir" -maxdepth 1 -type f -name '*.pcm' | wc -l | tr -d ' ')
    n_no_sr=$(find "$class_dir" -maxdepth 1 -type f -name '*.pcm' | { grep -vi 'SR[0-9]' || true; } | wc -l | tr -d ' ')
    size=$(du -sh "$class_dir" 2>/dev/null | cut -f1)
    printf "  %-14s %4s recordings  %6s\n" "$class_name" "$n_files" "$size"

    if [[ "$(echo "$class_name" | tr '[:upper:]' '[:lower:]')" == "noise" ]]; then
        has_noise=1
    elif (( n_files > 0 )); then
        n_classes=$((n_classes + 1))
    fi
    (( n_files == 0 )) && warn "class '$class_name' has no .pcm files"
    (( n_files > 0 && n_files < MIN_RECORDINGS )) && \
        warn "class '$class_name' has only $n_files recording(s); validation for it will be optimistic"
    (( n_no_sr > 0 )) && warn "$n_no_sr file(s) in '$class_name' have no SR<rate> tag in the name"
    n_files_total=$((n_files_total + n_files))
done < <(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

(( n_classes >= 2 )) || die "need at least 2 signal classes with recordings (found $n_classes)"
(( has_noise )) || info "No Noise/ folder: noise class will be built only from gaps in recordings"
info "$n_classes signal classes, $n_files_total recordings in total"

# ---- 3. run training -------------------------------------------------------
mkdir -p "$MODELS_DIR" "$LOGS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_FILE:-$MODELS_DIR/classifier_$TIMESTAMP.pt}"
LOG_FILE="$LOGS_DIR/train_$TIMESTAMP.log"

CMD=("$PYTHON" -u "$SCRIPT_DIR/train.py"
     --root "$DATASET_DIR"
     --out "$OUT_FILE"
     --epochs "$EPOCHS"
     --batch-size "$BATCH_SIZE"
     --workers "$WORKERS"
     --window-size "$WINDOW_SIZE"
     --stride "$STRIDE"
     ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

info "Log: $LOG_FILE"
info "Running: ${CMD[*]}"
echo "# ${CMD[*]}" > "$LOG_FILE"

START=$(date +%s)
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - START ))

# ---- 4. result -------------------------------------------------------------
if (( STATUS != 0 )); then
    die "train.py failed with exit code $STATUS after ${ELAPSED}s (see $LOG_FILE)"
fi
[[ -f "$OUT_FILE" ]] || die "train.py finished but no checkpoint at $OUT_FILE"

# Point models/latest.pt at the new checkpoint (copy if symlinks aren't supported)
if [[ "$(cd "$(dirname "$OUT_FILE")" && pwd)" == "$(cd "$MODELS_DIR" && pwd)" ]]; then
    ln -sfn "$(basename "$OUT_FILE")" "$MODELS_DIR/latest.pt" 2>/dev/null \
        || cp -f "$OUT_FILE" "$MODELS_DIR/latest.pt"
else
    cp -f "$OUT_FILE" "$MODELS_DIR/latest.pt"
fi

info "Done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
info "Checkpoint: $OUT_FILE"
info "Latest:     $MODELS_DIR/latest.pt"
info "Next step:  ./test.sh"
