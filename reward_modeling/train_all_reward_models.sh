#!/bin/bash
# train_all_reward_models.sh
#
# Train reward models for all criteria from preference data.
#
# Usage:
#   bash train_all_reward_models.sh

set -e

# =============================================================================
# Configuration
# =============================================================================
PREFERENCES_DIR="./preferences"
OUTPUT_DIR="./reward_models"
N_TRIALS=20
GPU_ID=0

# List of criteria to train (should match files in preferences dir)
# These correspond to the criteria in your config file
CRITERIA=(
    "empathy"
    "active_listening"
    "self_motivated_change"
    "trust_rapport"
    "patient_autonomy"
    "safety"
)

# =============================================================================
# Training Loop
# =============================================================================
echo "========================================"
echo "Training Reward Models for All Criteria"
echo "========================================"
echo "Preferences directory: $PREFERENCES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "HPO trials per criterion: $N_TRIALS"
echo ""

mkdir -p "$OUTPUT_DIR"

for criterion in "${CRITERIA[@]}"; do
    preferences_file="${PREFERENCES_DIR}/${criterion}_pairs.json"
    
    if [[ ! -f "$preferences_file" ]]; then
        echo "Warning: $preferences_file not found, skipping $criterion"
        continue
    fi
    
    echo "----------------------------------------"
    echo "Training reward model for: $criterion"
    echo "----------------------------------------"
    
    python train_reward_model.py \
        --preferences "$preferences_file" \
        --output_dir "${OUTPUT_DIR}/${criterion}" \
        --criterion "$criterion" \
        --n_trials "$N_TRIALS" \
        --gpu_id "$GPU_ID"
    
    echo "Completed: $criterion"
    echo ""
done

echo "========================================"
echo "All reward models trained!"
echo "========================================"
echo "Models saved to: $OUTPUT_DIR"
