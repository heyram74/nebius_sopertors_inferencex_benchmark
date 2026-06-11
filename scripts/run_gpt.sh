#!/usr/bin/env bash
# run_gptoss_h200.sh  —  Slurm wrapper for InferenceX gptoss_fp4_h200.sh
#
# Usage:
#   1. Edit the "Configure these" section below.
#   2. sbatch run_gptoss_h200.sh
#
# Prerequisites:
#   - Model weights pre-staged (run download_model.sh first)
#   - InferenceX repo cloned to $INFERENCEX_DIR
#   - enroot available on the cluster (standard on most NVIDIA-stack clusters)

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=inferencex-gptoss-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8            # full H200 SXM5 node
#SBATCH --time=02:00:00
#SBATCH --exclusive                  # ensures no other jobs share the node
#SBATCH --output=logs/gptoss_h200_%j.out
#SBATCH --error=logs/gptoss_h200_%j.err
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

# ── Configure these ───────────────────────────────────────────────────────────

# Absolute path to the pre-downloaded model weights (from download_model.sh).
# Must start with "/" so the benchmark script skips the in-job HF download.
MODEL="/home/models/openai--gpt-oss-120b"

# Absolute path to the InferenceX repo clone on shared storage.
INFERENCEX_DIR="/home/InferenceX"

# Where result JSONs will be written (must be writable inside the container).
#WORKSPACE="/home/results/gptoss_h200"

# vLLM Docker image. Pin to the same image used in InferenceX CI for H200.
# Check runners/launch_h200-*.sh in the repo for the latest pinned tag.
IMAGE="vllm/vllm-openai:v0.22.1"

# Path where the enroot squashfs image will be cached on shared storage.
# Avoids re-pulling on every run.
SQUASH_DIR="/home/containers"
SQUASH_FILE="${SQUASH_DIR}/vllm-v0.22.0.sqsh"

# ── Benchmark parameters ──────────────────────────────────────────────────────
export PORT=8000
export TP=2                    # tensor parallelism = number of GPUs
export CONC=64                 # max concurrent requests
export ISL=1024                # input sequence length (tokens)
export OSL=1024                # output sequence length (tokens)
export RANDOM_RANGE_RATIO=0.8  # 1.0 = fixed lengths, no variance
export RESULT_FILENAME="gptoss_fp4_h200_tp${TP}_conc${CONC}_isl${ISL}_osl${OSL}"
export RUN_EVAL=false          # set true to also run lm-eval accuracy checks
# ─────────────────────────────────────────────────────────────────────────────

# Where result JSONs will be written (must be writable inside the container).
WORKSPACE="/home/results/gptoss_h200_vllm22.0_sweep1/$RESULT_FILENAME"

echo "==> Job $SLURM_JOB_ID on node $SLURMD_NODENAME"
echo "    MODEL    : $MODEL"
echo "    TP       : $TP  CONC: $CONC  ISL: $ISL  OSL: $OSL"
echo "    WORKSPACE: $WORKSPACE"

mkdir -p "$WORKSPACE" "$SQUASH_DIR"

# ── Step 1: Pull the container image (only if not cached) ────────────────────
if [[ ! -f "$SQUASH_FILE" ]]; then
    echo "==> Importing Docker image to enroot squashfs (one-time, ~10 min)..."
    srun --exclusive \
         --time=30 \
         bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
else
    echo "==> Container image already cached at $SQUASH_FILE"
fi

# Create an rc script that sets env vars and runs the benchmark directly
RC_FILE=$(mktemp /tmp/enroot_rc_XXXXXX.sh)
cat > "$RC_FILE" <<EOF
#!/bin/bash
export MODEL="$MODEL"
export PORT="$PORT"
export TP="$TP"
export CONC="$CONC"
export ISL="$ISL"
export OSL="$OSL"
export RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO"
export RESULT_FILENAME="$RESULT_FILENAME"
export RUN_EVAL="$RUN_EVAL"
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_MXFP4_USE_MARLIN="1"

exec /bin/bash /inferencex/benchmarks/single_node/fixed_seq_len/gptoss_fp4_h200.sh
EOF
chmod +x "$RC_FILE"

# ── Step 2: Run the benchmark inside the container ───────────────────────────
echo "==> Launching benchmark..."
srun --exclusive \
     --time=120 \
     --gpus-per-node=8 \
     enroot start \
         --rc "$RC_FILE" \
         --mount "${INFERENCEX_DIR}:/inferencex" \
         --mount "${MODEL}:${MODEL}" \
         --mount "${WORKSPACE}:/workspace" \
         --rw \
         "$SQUASH_FILE"

rm -f "$RC_FILE"

echo "==> Benchmark finished. Results in $WORKSPACE"
ls -lh "${WORKSPACE}/"*.json 2>/dev/null || echo "  (no JSON files found — check logs)"
