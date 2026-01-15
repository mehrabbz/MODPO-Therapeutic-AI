"""
Generate Evaluation Responses

Generates therapeutic responses from trained models for evaluation.
Supports loading LoRA adapters, base models, and merged models (DPO Soup).

Usage:
    # Single model
    python generate_eval_responses.py \
        --model_path ./models/modpo_empathy/final_model \
        --model_name modpo_empathy \
        --test_questions ./data/questions_test.csv \
        --output_dir ./evaluation_responses

    # Base model (no adapter)
    python generate_eval_responses.py \
        --model_name base \
        --test_questions ./data/questions_test.csv \
        --output_dir ./evaluation_responses

    # Merged model (DPO Soup)
    python generate_eval_responses.py \
        --model_path ./models/dpo_empathy/final_model \
        --merge_with ./models/dpo_safety/final_model \
        --merge_weights 0.5 0.5 \
        --model_name dpo_soup_empathy_safety \
        --test_questions ./data/questions_test.csv \
        --output_dir ./evaluation_responses
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation responses from trained models")
    
    # Model specification
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained LoRA model. If not provided, uses base model only.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name for this model (used in output filename)")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model name")
    
    # Merging options (for DPO Soup)
    parser.add_argument("--merge_with", type=str, default=None,
                        help="Path to second model for merging (DPO Soup)")
    parser.add_argument("--merge_weights", type=float, nargs=2, default=[0.5, 0.5],
                        help="Weights for merging [model1_weight, model2_weight]")
    
    # Input/Output
    parser.add_argument("--test_questions", type=str, required=True,
                        help="Path to test questions CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for responses")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    
    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (1 recommended for consistency)")
    
    # Options
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output file already exists")
    
    return parser.parse_args()


def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def load_base_model(model_name: str, device: str):
    """Load base model with half precision."""
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    return model


def load_lora_model(base_model, adapter_path: str):
    """Load LoRA adapter onto base model."""
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def merge_lora_models(base_model, model_path_1: str, model_path_2: str, 
                      weight_1: float = 0.5, weight_2: float = 0.5):
    """
    Merge two LoRA models using weighted parameter averaging.
    
    This implements the DPO Soup approach from:
    Jang et al., "Personalized Soups" (arXiv:2310.11564)
    """
    logger.info(f"Merging models with weights: {weight_1:.2f}, {weight_2:.2f}")
    
    # Load first adapter and get its weights
    model_1 = PeftModel.from_pretrained(base_model, model_path_1)
    lora_weights_1 = get_peft_model_state_dict(model_1)
    
    # Scale the weights
    scaled_weights_1 = {k: v * weight_1 for k, v in lora_weights_1.items()}
    
    # Set scaled weights and merge into base
    set_peft_model_state_dict(model_1, scaled_weights_1)
    merged_model = model_1.merge_and_unload()
    logger.info("First adapter merged")
    
    # Load second adapter on merged model
    model_2 = PeftModel.from_pretrained(merged_model, model_path_2)
    lora_weights_2 = get_peft_model_state_dict(model_2)
    
    # Scale second adapter weights
    scaled_weights_2 = {k: v * weight_2 for k, v in lora_weights_2.items()}
    
    # Set scaled weights and merge
    set_peft_model_state_dict(model_2, scaled_weights_2)
    final_model = model_2.merge_and_unload()
    logger.info("Second adapter merged - DPO Soup complete")
    
    return final_model


def generate_response(model, tokenizer, question: str, 
                      temperature: float = 0.8, 
                      max_new_tokens: int = 512,
                      top_p: float = 0.95) -> str:
    """Generate a therapeutic response for a question."""
    
    # Format as chat
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0"
    
    logger.info(f"Using GPU {args.gpu_id}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if output exists
    output_file = output_dir / f"{args.model_name}_responses.csv"
    if args.skip_existing and output_file.exists():
        logger.info(f"Output file exists, skipping: {output_file}")
        return
    
    # Load test questions
    logger.info(f"Loading test questions from: {args.test_questions}")
    test_df = pd.read_csv(args.test_questions)
    
    # Determine question column name
    if "question_text" in test_df.columns:
        question_col = "question_text"
    elif "question" in test_df.columns:
        question_col = "question"
    else:
        question_col = test_df.columns[0]
        logger.warning(f"Using first column as question: {question_col}")
    
    # Determine ID column
    if "question_id" in test_df.columns:
        id_col = "question_id"
    else:
        test_df["question_id"] = [f"q_{i}" for i in range(len(test_df))]
        id_col = "question_id"
    
    logger.info(f"Loaded {len(test_df)} test questions")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    try:
        base_model = load_base_model(args.base_model, device)
        
        if args.merge_with:
            # DPO Soup: merge two models
            if not args.model_path:
                raise ValueError("--model_path required when using --merge_with")
            model = merge_lora_models(
                base_model,
                args.model_path,
                args.merge_with,
                args.merge_weights[0],
                args.merge_weights[1]
            )
        elif args.model_path:
            # Single LoRA adapter
            model = load_lora_model(base_model, args.model_path)
        else:
            # Base model only
            model = base_model
            logger.info("Using base model without adapter")
        
        logger.info(f"Model loaded successfully: {args.model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Generate responses
    responses_data = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Generating ({args.model_name})"):
        question_id = row[id_col]
        question = row[question_col]
        
        try:
            response = generate_response(
                model, tokenizer, question,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p
            )
        except Exception as e:
            logger.warning(f"Error generating response for {question_id}: {e}")
            response = f"ERROR: {str(e)}"
        
        responses_data.append({
            "question_id": question_id,
            "question_text": question,
            "response_text": response,
            "model_name": args.model_name
        })
    
    # Save responses
    responses_df = pd.DataFrame(responses_data)
    responses_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(responses_df)} responses to: {output_file}")
    
    # Cleanup
    del model
    if args.model_path:
        del base_model
    clear_gpu_memory()
    
    print(f"\n{'='*60}")
    print(f"Response Generation Complete!")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Questions: {len(test_df)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
