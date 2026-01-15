#!/bin/bash
# run_parallel_generation.sh - Multi-GPU Parallel Response Generation
#
# Usage (run from response_generation folder):
#   bash run_parallel_generation.sh
#
# Configure the variables below before running.

set -e

# =============================================================================
# Configuration - EDIT THESE
# =============================================================================
INPUT_FILE="./processed/questions_train.csv"
OUTPUT_DIR="./responses"
NUM_GPUS=4                    # Number of GPUs to use
GPU_IDS=(0 1 2 3)             # Specific GPU IDs (adjust based on your setup)
BATCH_SIZE=4                  # Questions per batch per GPU
GEN_BATCH_SIZE=16             # Generation batch size
NUM_RESPONSES=5               # Responses per question
TEMPERATURE=0.8
RESUME=true                   # Set to false for fresh start

# =============================================================================
# Script Logic - Don't edit below unless needed
# =============================================================================

echo "========================================"
echo "Multi-GPU Parallel Response Generation"
echo "========================================"

# Validate input file
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "chunks"

echo "Configuration:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: ${GPU_IDS[*]}"
echo "  Responses per question: $NUM_RESPONSES"
echo ""

# Split data into chunks
echo "Splitting dataset into $NUM_GPUS chunks..."
python3 -c "
import pandas as pd
import math

df = pd.read_csv('$INPUT_FILE')
total = len(df)
chunk_size = math.ceil(total / $NUM_GPUS)

print(f'Total questions: {total}')
print(f'Chunk size: {chunk_size}')

for i in range($NUM_GPUS):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, total)
    if start < total:
        chunk = df.iloc[start:end]
        chunk.to_csv(f'chunks/chunk_{i}.csv', index=False)
        print(f'Created chunks/chunk_{i}.csv: {len(chunk)} questions')
"

echo ""
echo "Launching GPU processes..."

# Launch processes
for idx in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$idx]}
    chunk_file="chunks/chunk_${idx}.csv"
    
    if [[ ! -f "$chunk_file" ]]; then
        echo "Skipping GPU $gpu_id (no chunk file)"
        continue
    fi
    
    cmd="python3 generate_responses_parallel.py $chunk_file \
        --output_dir $OUTPUT_DIR \
        --gpu_id $gpu_id \
        --batch_size $BATCH_SIZE \
        --generation_batch_size $GEN_BATCH_SIZE \
        --num_responses $NUM_RESPONSES \
        --temperature $TEMPERATURE"
    
    if [[ "$RESUME" == "true" ]]; then
        cmd="$cmd --resume"
    fi
    
    echo "Starting GPU $gpu_id..."
    eval $cmd > "${OUTPUT_DIR}/gpu_${gpu_id}.log" 2>&1 &
    echo $! > "${OUTPUT_DIR}/gpu_${gpu_id}.pid"
    
    sleep 2
done

echo ""
echo "All processes launched. Monitoring..."
echo "Check logs: tail -f ${OUTPUT_DIR}/gpu_*.log"
echo ""

# Monitor until complete
while true; do
    active=0
    for gpu_id in "${GPU_IDS[@]}"; do
        pid_file="${OUTPUT_DIR}/gpu_${gpu_id}.pid"
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                ((active++))
            else
                rm -f "$pid_file"
            fi
        fi
    done
    
    if [[ $active -eq 0 ]]; then
        echo "All processes complete!"
        break
    fi
    
    echo "Active processes: $active"
    sleep 30
done

# Combine results
echo ""
echo "Combining results..."
python3 -c "
import pandas as pd
import glob

dfs = []
for f in glob.glob('$OUTPUT_DIR/gpu_*/*_responses.csv'):
    dfs.append(pd.read_csv(f))
    print(f'Loaded: {f}')

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['question_id', 'response_id'])
    combined.to_csv('$OUTPUT_DIR/responses_combined.csv', index=False)
    print(f'Combined: {len(combined)} responses')
    print(f'Unique questions: {combined[\"question_id\"].nunique()}')
"

# Cleanup
rm -rf chunks/
rm -f ${OUTPUT_DIR}/gpu_*.pid

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}/responses_combined.csv"
