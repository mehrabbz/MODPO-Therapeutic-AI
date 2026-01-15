#!/bin/bash
# Generate evaluation responses from all trained models
# 
# Usage:
#   ./generate_all_responses.sh [GPU_ID]
#
# Example:
#   ./generate_all_responses.sh 0

GPU_ID=${1:-0}
TEST_QUESTIONS="./data/questions_test.csv"
OUTPUT_DIR="./evaluation_responses"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

echo "========================================"
echo "Generating Evaluation Responses"
echo "GPU: $GPU_ID"
echo "Test Questions: $TEST_QUESTIONS"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# Base model (no adapter)
echo -e "\n>>> Generating: base_model"
python generate_eval_responses.py \
    --model_name base_model \
    --test_questions $TEST_QUESTIONS \
    --output_dir $OUTPUT_DIR \
    --gpu_id $GPU_ID \
    --skip_existing

# SFT models
for CRITERION in empathy safety active_listening self_motivated_change trust_rapport patient_autonomy; do
    MODEL_PATH="./models/sft_${CRITERION}/final_model"
    if [ -d "$MODEL_PATH" ]; then
        echo -e "\n>>> Generating: sft_${CRITERION}"
        python generate_eval_responses.py \
            --model_path $MODEL_PATH \
            --model_name "sft_${CRITERION}" \
            --test_questions $TEST_QUESTIONS \
            --output_dir $OUTPUT_DIR \
            --gpu_id $GPU_ID \
            --skip_existing
    fi
done

# DPO models
for CRITERION in empathy safety active_listening self_motivated_change trust_rapport patient_autonomy; do
    MODEL_PATH="./models/dpo_${CRITERION}/final_model"
    if [ -d "$MODEL_PATH" ]; then
        echo -e "\n>>> Generating: dpo_${CRITERION}"
        python generate_eval_responses.py \
            --model_path $MODEL_PATH \
            --model_name "dpo_${CRITERION}" \
            --test_questions $TEST_QUESTIONS \
            --output_dir $OUTPUT_DIR \
            --gpu_id $GPU_ID \
            --skip_existing
    fi
done

# MODPO models
for CRITERION in empathy safety; do
    MODEL_PATH="./models/modpo_${CRITERION}/final_model"
    if [ -d "$MODEL_PATH" ]; then
        echo -e "\n>>> Generating: modpo_${CRITERION}"
        python generate_eval_responses.py \
            --model_path $MODEL_PATH \
            --model_name "modpo_${CRITERION}" \
            --test_questions $TEST_QUESTIONS \
            --output_dir $OUTPUT_DIR \
            --gpu_id $GPU_ID \
            --skip_existing
    fi
done

# Joint-Loss DPO
MODEL_PATH="./models/joint_loss_dpo/final_model"
if [ -d "$MODEL_PATH" ]; then
    echo -e "\n>>> Generating: joint_loss_dpo"
    python generate_eval_responses.py \
        --model_path $MODEL_PATH \
        --model_name "joint_loss_dpo" \
        --test_questions $TEST_QUESTIONS \
        --output_dir $OUTPUT_DIR \
        --gpu_id $GPU_ID \
        --skip_existing
fi

# DPO Soup (merged models)
DPO_EMPATHY="./models/dpo_empathy/final_model"
DPO_SAFETY="./models/dpo_safety/final_model"
if [ -d "$DPO_EMPATHY" ] && [ -d "$DPO_SAFETY" ]; then
    echo -e "\n>>> Generating: dpo_soup_empathy_safety"
    python generate_eval_responses.py \
        --model_path $DPO_EMPATHY \
        --merge_with $DPO_SAFETY \
        --merge_weights 0.5 0.5 \
        --model_name "dpo_soup_empathy_safety" \
        --test_questions $TEST_QUESTIONS \
        --output_dir $OUTPUT_DIR \
        --gpu_id $GPU_ID \
        --skip_existing
fi

# MODPO Survey models
for VARIANT in survey survey4 maxim; do
    MODEL_PATH="./models/modpo_${VARIANT}/final_model"
    if [ -d "$MODEL_PATH" ]; then
        echo -e "\n>>> Generating: modpo_${VARIANT}"
        python generate_eval_responses.py \
            --model_path $MODEL_PATH \
            --model_name "modpo_${VARIANT}" \
            --test_questions $TEST_QUESTIONS \
            --output_dir $OUTPUT_DIR \
            --gpu_id $GPU_ID \
            --skip_existing
    fi
done

echo -e "\n========================================"
echo "Response generation complete!"
echo "Output files in: $OUTPUT_DIR"
echo "========================================"
ls -la $OUTPUT_DIR/*.csv 2>/dev/null || echo "No response files found"
