#!/usr/bin/bash
#
# Run eynollah layout analysis over a list of images, in mini-batches.
#
# Usage: layout_analysis.sh <path_to_file_list>
#
# The file list holds one image path per line, e.g.
#     ptb/1889/ptb_18891212/ptb_18891212_027.jpg
# Relative paths are resolved against $IMAGE_ROOT.
#
# Environment (set by the sbatch scripts; defaults in brackets):
#   CONDA_PREFIX   eynollah env prefix                       [required]
#   IMAGE_ROOT     dir the relative list paths live under    [current dir]
#   OUTPUT_ROOT    results go to $OUTPUT_ROOT/<name>/<year>  [required]
#   MODEL_DIR      eynollah model directory                  [model_dir]
#   BATCH_SIZE     images per eynollah invocation            [200]
#   SKIP_DONE      1 = skip images whose .xml already exists [1]

# Configuration
BATCH_SIZE="${BATCH_SIZE:-200}"
MODEL_DIR="${MODEL_DIR:-model_dir}"
SKIP_DONE="${SKIP_DONE:-1}"

# Accept file_list as first argument (no interactive read)
file_list="$1"

if [ -z "$file_list" ]; then
    echo "Usage: $0 <path_to_file_list>"
    exit 1
fi

if [ ! -f "$file_list" ]; then
    echo "Error: File list not found at $file_list"
    exit 1
fi

if [ -z "$OUTPUT_ROOT" ]; then
    echo "ERROR: OUTPUT_ROOT not set. Export it (results go to \$OUTPUT_ROOT/<name>/<year>)."
    exit 1
fi

# ---------------------------------------------------------
# 0. CUDA / cuDNN ENVIRONMENT
# ---------------------------------------------------------
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: CONDA_PREFIX not set. The caller must export it (pointing to the eynollah env)."
    exit 1
fi

# TF 2.12 needs CUDA 11.8 / cuDNN 8, which the conda env provides in $CONDA_PREFIX/lib.
# (No pip nvidia.cudnn path here: in the new env that is cuDNN 9 for CUDA 13 -> wrong version.)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -n "$MAMBA_ROOT_PREFIX" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib"
fi

# Several processes share one GPU: TensorFlow must NOT grab all VRAM up front.
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

# Optional GPU sanity check on first run (comment out once stable)
# python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# ---------------------------------------------------------
# 1. SETUP UNIQUE TEMP FILES
# ---------------------------------------------------------
base_filename=$(basename "$file_list")
base_name="${base_filename%.*}"
batch_file="batch_${base_name}_$$.txt"
work_list="${batch_file}.remaining"

echo "Using temporary batch file: $batch_file"

trap 'rm -f "$batch_file" "$work_list"' EXIT

# ---------------------------------------------------------
# 2. DETERMINE OUTPUT DIRECTORY
# ---------------------------------------------------------
first_input_path=$(grep -m 1 "." "$file_list")
if [ -z "$first_input_path" ]; then
    echo "Error: The file list appears to be empty."
    exit 1
fi

name=$(echo "$first_input_path" | awk -F/ '{print $(NF-3)}')
year=$(echo "$first_input_path" | awk -F/ '{print $(NF-2)}')
output_dir="$OUTPUT_ROOT/$name/$year"

echo "Output directory: $output_dir"
mkdir -p "$output_dir"

# ---------------------------------------------------------
# 3. BUILD THE WORK LIST
#    - drop blank lines and Windows line endings
#    - make relative paths absolute under IMAGE_ROOT
#    - optionally skip images that already have an output XML (resume)
# ---------------------------------------------------------
# Work on a copy so we don't destructively modify the user's input list
tr -d '\r' < "$file_list" | grep -v '^[[:space:]]*$' \
  | awk -v root="${IMAGE_ROOT:-}" -v out="$output_dir" -v skip="$SKIP_DONE" '
        {
            p = $0
            if (root != "" && substr(p, 1, 1) != "/") p = root "/" p
            if (skip == "1") {
                f = p; sub(/.*\//, "", f); sub(/\.[^.]*$/, "", f)
                if ((getline _ < (out "/" f ".xml")) > 0) { close(out "/" f ".xml"); done++; next }
            }
            print p
        }
        END { if (done) print "Skipping " done " already processed image(s)" > "/dev/stderr" }
    ' > "$work_list"

total=$(wc -l < "$work_list")
echo "$total image(s) to process for $name/$year"

# ---------------------------------------------------------
# 4. THE MINI-BATCH LOOP
# ---------------------------------------------------------
while [ -s "$work_list" ]; do

    head -n "$BATCH_SIZE" "$work_list" > "$batch_file"

    count=$(wc -l < "$batch_file")
    now=$(date +"%R")
    echo "---------------------------------------------------"
    echo "$now [$name/$year]: Processing batch of $count files..."

    if ! eynollah -m "$MODEL_DIR" layout -i "$batch_file" -o "$output_dir" -cl -fl; then
        echo "CRITICAL ERROR: eynollah crashed."
        echo "The problematic batch is preserved in: $PWD/$batch_file"
        trap - EXIT
        rm -f "$work_list"
        exit 1
    fi

    sed -i "1,${BATCH_SIZE}d" "$work_list"
done

echo "Processing complete: $name/$year"