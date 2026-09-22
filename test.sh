#!/usr/bin/env bash
#
# Test sequence for the radio-standard classifier.
#
# Evaluation mode (default): runs the model on every class folder in the test
# set, compares each verdict with the folder name, and prints per-class and
# overall recording-level accuracy plus a list of misclassified files.
#
# Ad-hoc mode: pass files or folders as arguments to just classify them
# (no scoring).
#
#   1. checks the Python environment and the checkpoint
#   2. warns if test files also appear in the training dataset (leakage)
#   3. runs test_model.py, logging to logs/test_<timestamp>.log
#
# Usage:
#   ./test.sh                                  # evaluate TestSet/ with models/latest.pt
#   ./test.sh -c models/classifier_X.pt -t TestSet -r 0.6
#   ./test.sh -l some_recording_SR100446.pcm   # ad-hoc, with per-window timeline
#   ./test.sh -- --sample-rate 100446          # anything after -- goes to test_model.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- defaults (override with options or environment variables) -------------
PYTHON="${PYTHON:-python3}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/models/latest.pt}"
TEST_DIR="${TEST_DIR:-$SCRIPT_DIR/TestSet}"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/Dataset}"
LOGS_DIR="${LOGS_DIR:-$SCRIPT_DIR/logs}"
THRESHOLD="${THRESHOLD:-0.0}"

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF
Options:
  -c FILE   checkpoint                  (default: $CHECKPOINT)
  -t DIR    test set folder             (default: $TEST_DIR)
  -D DIR    training dataset, for the leakage check (default: $DATASET_DIR)
  -r X      'unknown' below this confidence (default: $THRESHOLD)
  -l        print per-window timeline
  -x        force CPU
  -h        show this help
EOF
}

TIMELINE=0
FORCE_CPU=0
while getopts ":c:t:D:r:lxh" opt; do
    case "$opt" in
        c) CHECKPOINT="$OPTARG" ;;
        t) TEST_DIR="$OPTARG" ;;
        D) DATASET_DIR="$OPTARG" ;;
        r) THRESHOLD="$OPTARG" ;;
        l) TIMELINE=1 ;;
        x) FORCE_CPU=1 ;;
        h) usage; exit 0 ;;
        :) echo "Option -$OPTARG needs a value" >&2; exit 2 ;;
        \?) echo "Unknown option -$OPTARG (use -h)" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

# Positional args before "--" are inputs; everything after "--" goes to test_model.py
INPUTS=()
EXTRA_ARGS=()
while (( $# )); do
    if [[ "$1" == "--" ]]; then shift; EXTRA_ARGS=("$@"); break; fi
    INPUTS+=("$1"); shift
done

info() { echo "[test.sh] $*"; }
warn() { echo "[test.sh] WARNING: $*" >&2; }
die()  { echo "[test.sh] ERROR: $*" >&2; exit 1; }
lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

# ---- 1. environment and checkpoint -----------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || die "'$PYTHON' not found (set PYTHON=...)"
for f in common.py test_model.py; do
    [[ -f "$SCRIPT_DIR/$f" ]] || die "$f not found next to this script"
done
[[ -e "$CHECKPOINT" ]] || die "checkpoint '$CHECKPOINT' not found (run ./train.sh first, or use -c)"

MODEL_CLASSES="$("$PYTHON" - "$CHECKPOINT" <<'EOF'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(" ".join(ckpt["class_names"]))
EOF
)" || die "could not read checkpoint (missing packages? pip install numpy scipy torch)"
info "Checkpoint: $CHECKPOINT"
info "Model classes: $MODEL_CLASSES"

COMMON_ARGS=(--checkpoint "$CHECKPOINT" --threshold "$THRESHOLD")
(( TIMELINE )) && COMMON_ARGS+=(--timeline)
(( FORCE_CPU )) && COMMON_ARGS+=(--cpu)
COMMON_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/test_$(date +%Y%m%d_%H%M%S).log"
n=1; base="${LOG_FILE%.log}"
while [[ -e "$LOG_FILE" ]]; do n=$((n + 1)); LOG_FILE="${base}_$n.log"; done
info "Log: $LOG_FILE"
: > "$LOG_FILE"

run_model() {  # run test_model.py on the given inputs, log and echo the output
    "$PYTHON" -u "$SCRIPT_DIR/test_model.py" "$@" "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

# ---- ad-hoc mode -------------------------------------------------------------
if (( ${#INPUTS[@]} > 0 )); then
    run_model "${INPUTS[@]}" || die "test_model.py failed (see $LOG_FILE)"
    exit 0
fi

# ---- evaluation mode ---------------------------------------------------------
[[ -d "$TEST_DIR" ]] || die "test set folder '$TEST_DIR' does not exist (or pass files as arguments)"
info "Test set: $TEST_DIR"

# 2. leakage check: same file name in the training dataset?
if [[ -d "$DATASET_DIR" ]]; then
    dup=$(comm -12 \
        <(find "$DATASET_DIR" -type f -name '*.pcm' | sed 's#.*/##' | sort -u) \
        <(find "$TEST_DIR"    -type f -name '*.pcm' | sed 's#.*/##' | sort -u))
    if [[ -n "$dup" ]]; then
        warn "these test files also exist in $DATASET_DIR (results will be optimistic):"
        echo "$dup" | sed 's/^/    /' >&2
    fi
fi

# Extract "file<TAB>verdict" pairs from test_model.py output
parse_verdicts() {
    awk '
        /^=== / { f = $0; sub(/^=== /, "", f); sub(/  \([0-9]+ windows\)$/, "", f); next }
        /^  -> no signal detected/ { print f "\t" "__noise__"; next }
        /^  -> no active windows/  { print f "\t" "__none__";  next }
        /^  -> /                   { print f "\t" $2 }
    '
}

TOTAL=0
CORRECT=0
SUMMARY=""
ERRORS=""

while IFS= read -r class_dir; do
    class_name="$(basename "$class_dir")"
    n_files=$(find "$class_dir" -type f -name '*.pcm' | wc -l | tr -d ' ')
    (( n_files > 0 )) || continue

    if [[ " $(lower "$MODEL_CLASSES") " != *" $(lower "$class_name") "* ]]; then
        warn "test folder '$class_name' is not a class of this model; all its files will count as errors"
    fi

    echo | tee -a "$LOG_FILE"
    info "==== Class $class_name ($n_files recordings) ====" | tee -a "$LOG_FILE"
    # tee to stderr so the output is shown live while also being captured
    output="$(run_model "$class_dir" | tee /dev/stderr)" || die "test_model.py failed on $class_dir (see $LOG_FILE)"

    c_total=0; c_correct=0
    while IFS=$'\t' read -r file verdict; do
        [[ -z "$file" ]] && continue
        c_total=$((c_total + 1))
        if [[ "$verdict" == "__noise__" ]]; then
            shown="noise"; ok=$([[ "$(lower "$class_name")" == "noise" ]] && echo 1 || echo 0)
        else
            shown="$verdict"; ok=$([[ "$(lower "$verdict")" == "$(lower "$class_name")" ]] && echo 1 || echo 0)
        fi
        if (( ok )); then
            c_correct=$((c_correct + 1))
        else
            ERRORS+="$(printf '    %-10s -> %-10s %s' "$class_name" "$shown" "$file")"$'\n'
        fi
    done < <(echo "$output" | parse_verdicts)

    TOTAL=$((TOTAL + c_total)); CORRECT=$((CORRECT + c_correct))
    if (( c_total > 0 )); then
        pct=$(( 100 * c_correct / c_total ))
        SUMMARY+="$(printf '  %-14s %3d / %-3d  (%3d%%)' "$class_name" "$c_correct" "$c_total" "$pct")"$'\n'
    fi
done < <(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Loose files directly in TEST_DIR have no class folder: classify, don't score
loose=()
while IFS= read -r f; do loose+=("$f"); done < <(find "$TEST_DIR" -maxdepth 1 -type f -name '*.pcm' | sort)
if (( ${#loose[@]} > 0 )); then
    echo | tee -a "$LOG_FILE"
    info "==== Unlabelled files in $TEST_DIR (not scored) ====" | tee -a "$LOG_FILE"
    run_model "${loose[@]}" || warn "test_model.py failed on unlabelled files"
fi

(( TOTAL > 0 )) || die "no labelled recordings found in class folders of $TEST_DIR"

{
    echo
    echo "================ SUMMARY ================"
    echo "Checkpoint: $CHECKPOINT"
    echo "Recording-level accuracy per class:"
    printf '%s' "$SUMMARY"
    echo "-----------------------------------------"
    printf '  %-14s %3d / %-3d  (%3d%%)\n' "OVERALL" "$CORRECT" "$TOTAL" $(( 100 * CORRECT / TOTAL ))
    if [[ -n "$ERRORS" ]]; then
        echo
        echo "Misclassified (true -> predicted, file):"
        printf '%s' "$ERRORS"
    fi
} | tee -a "$LOG_FILE"
