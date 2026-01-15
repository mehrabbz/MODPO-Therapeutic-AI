#!/bin/bash
# Run head-to-head evaluation for all model pairs
#
# Usage:
#   ./run_all_evaluations.sh [--resume]
#
# This script:
# 1. Finds all model response files
# 2. Generates all pairwise combinations
# 3. Runs evaluations in parallel (configurable)

set -e

# Configuration
RESPONSES_DIR="./evaluation_responses"
PERSONAS_FILE="./personas/test_personas.json"
OUTPUT_DIR="./evaluation_results"
CRITERIA="empathy safety overall_preference"
NUM_PERSONAS=50
PARALLEL_JOBS=4  # Number of parallel evaluations

# Parse arguments
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "Will resume from checkpoints"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "./logs"

# Find all model response files
echo "Finding model response files..."
MODEL_FILES=$(ls "$RESPONSES_DIR"/*_responses.csv 2>/dev/null || true)

if [[ -z "$MODEL_FILES" ]]; then
    echo "ERROR: No response files found in $RESPONSES_DIR"
    exit 1
fi

# Extract model names
MODELS=()
for f in $MODEL_FILES; do
    model_name=$(basename "$f" | sed 's/_responses.csv//')
    MODELS+=("$model_name")
done

echo "Found ${#MODELS[@]} models: ${MODELS[*]}"

# Generate all pairs
echo "Generating model pairs..."
PAIRS=()
for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
        PAIRS+=("${MODELS[$i]}:${MODELS[$j]}")
    done
done

echo "Total pairs to evaluate: ${#PAIRS[@]}"

# Function to run single evaluation
run_evaluation() {
    local pair=$1
    local model1=$(echo "$pair" | cut -d: -f1)
    local model2=$(echo "$pair" | cut -d: -f2)
    local log_file="./logs/${model1}_vs_${model2}.log"
    
    echo "Starting: $model1 vs $model2"
    
    python evaluate_head_to_head.py \
        --model1 "$model1" \
        --model2 "$model2" \
        --responses_dir "$RESPONSES_DIR" \
        --personas "$PERSONAS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --criteria $CRITERIA \
        --num_personas $NUM_PERSONAS \
        $RESUME_FLAG \
        > "$log_file" 2>&1
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Completed: $model1 vs $model2"
    else
        echo "❌ Failed: $model1 vs $model2 (see $log_file)"
    fi
}

export -f run_evaluation
export RESPONSES_DIR PERSONAS_FILE OUTPUT_DIR CRITERIA NUM_PERSONAS RESUME_FLAG

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with $PARALLEL_JOBS jobs"
    printf '%s\n' "${PAIRS[@]}" | parallel -j $PARALLEL_JOBS run_evaluation {}
else
    echo "GNU parallel not found, running sequentially"
    for pair in "${PAIRS[@]}"; do
        run_evaluation "$pair"
    done
fi

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results in: $OUTPUT_DIR"
echo "========================================"

# Print summary
echo ""
echo "Summary of results:"
for pair in "${PAIRS[@]}"; do
    model1=$(echo "$pair" | cut -d: -f1)
    model2=$(echo "$pair" | cut -d: -f2)
    summary_file="$OUTPUT_DIR/${model1}_vs_${model2}/summary_stats.json"
    
    if [[ -f "$summary_file" ]]; then
        echo ""
        echo "--- $model1 vs $model2 ---"
        python3 -c "
import json
with open('$summary_file') as f:
    s = json.load(f)
for criterion in s.get('criteria', []):
    if criterion in s:
        stats = s[criterion]
        m1_wr = stats.get('${model1}_winrate', 0)
        m2_wr = stats.get('${model2}_winrate', 0)
        print(f'  {criterion}: ${model1} {m1_wr:.1f}% vs ${model2} {m2_wr:.1f}%')
"
    fi
done
