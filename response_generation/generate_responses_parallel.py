"""
Parallel Therapeutic Response Generator (Multi-GPU)

Optimized for multi-GPU setups with checkpoint/resume functionality.
For single-GPU usage, see `generate_responses.py`.

Usage:
    # Single GPU with specific ID
    python generate_responses_parallel.py ./processed/questions_train.csv \
        --output_dir ./responses \
        --gpu_id 0 \
        --num_responses 5

    # With resume functionality
    python generate_responses_parallel.py ./processed/questions_train.csv \
        --output_dir ./responses \
        --gpu_id 0 \
        --resume

For running on multiple GPUs simultaneously, use the provided bash script:
    bash scripts/run_parallel_generation.sh
"""

import argparse
import glob
import logging
import os
import warnings
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel therapy response generator with resume functionality"
    )
    parser.add_argument(
        "input_file",
        help="Input CSV file with questions"
    )
    parser.add_argument(
        "--output_dir",
        default="responses",
        help="Output directory"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Questions per batch"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=16,
        help="Generation batch size"
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Responses per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model name"
    )
    return parser.parse_args()


class TherapyResponseGenerator:
    """Parallel therapy response generator optimized for multi-GPU setups."""
    
    def __init__(
        self, 
        model_name: str,
        gpu_id: int = 0, 
        generation_batch_size: int = 16
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        self.generation_batch_size = generation_batch_size
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        print(f"[GPU {gpu_id}] Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        print(f"[GPU {gpu_id}] Model loaded successfully")
    
    def generate_responses_batch(
        self, 
        questions: List[str], 
        num_responses: int = 5, 
        temperature: float = 0.8,
        max_new_tokens: int = 512
    ) -> List[str]:
        """Generate multiple responses for multiple questions efficiently."""
        all_prompts = []
        
        # Create all prompts
        for question in questions:
            prompt = f"[INST] You are a therapist. Respond to this person's question:\n{question} [/INST]"
            all_prompts.extend([prompt] * num_responses)
        
        all_responses = []
        
        # Process in smaller batches
        for i in tqdm(
            range(0, len(all_prompts), self.generation_batch_size),
            desc=f"GPU {self.gpu_id} Generating",
            leave=False
        ):
            batch_prompts = all_prompts[i:i + self.generation_batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            for output in outputs:
                generated_tokens = output[inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                ).strip()
                all_responses.append(response)
            
            del inputs, outputs
            torch.cuda.empty_cache()
        
        return all_responses


def load_existing_progress(output_dir: str, gpu_id: int):
    """Load existing progress for resume functionality."""
    gpu_output_dir = f"{output_dir}/gpu_{gpu_id}"
    
    if not os.path.exists(gpu_output_dir):
        return None, set()
    
    # Look for completed or checkpoint files
    response_files = glob.glob(f"{gpu_output_dir}/*_responses.csv")
    checkpoint_files = glob.glob(f"{gpu_output_dir}/checkpoint_*.csv")
    
    existing_responses = None
    
    if response_files:
        latest_file = max(response_files, key=os.path.getctime)
        existing_responses = pd.read_csv(latest_file)
        print(f"[GPU {gpu_id}] Found completed responses: {latest_file}")
    elif checkpoint_files:
        latest_checkpoint = max(
            checkpoint_files, 
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        existing_responses = pd.read_csv(latest_checkpoint)
        print(f"[GPU {gpu_id}] Found checkpoint: {latest_checkpoint}")
    
    if existing_responses is not None:
        processed_ids = set(existing_responses['question_id'].unique())
        print(f"[GPU {gpu_id}] Found {len(processed_ids)} processed questions")
        return existing_responses, processed_ids
    
    return None, set()


def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    gpu_output_dir = f"{args.output_dir}/gpu_{args.gpu_id}"
    os.makedirs(gpu_output_dir, exist_ok=True)
    
    print(f"[GPU {args.gpu_id}] Starting therapy response generation")
    
    # Load data
    print(f"[GPU {args.gpu_id}] Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    original_count = len(df)
    print(f"[GPU {args.gpu_id}] Loaded {original_count} questions")
    
    # Handle resume
    all_responses_data = []
    if args.resume:
        existing_responses, processed_ids = load_existing_progress(
            args.output_dir, 
            args.gpu_id
        )
        if existing_responses is not None:
            all_responses_data = existing_responses.to_dict('records')
            df = df[~df['question_id'].isin(processed_ids)]
            print(f"[GPU {args.gpu_id}] Remaining: {len(df)} questions")
    
    if len(df) == 0:
        print(f"[GPU {args.gpu_id}] No questions to process. Complete!")
        return
    
    # Initialize generator
    generator = TherapyResponseGenerator(
        args.model_name,
        args.gpu_id, 
        args.generation_batch_size
    )
    
    # Process in batches
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    print(f"[GPU {args.gpu_id}] Processing {total_batches} batches...")
    
    for batch_idx in tqdm(range(total_batches), desc=f"GPU {args.gpu_id} Batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        questions = batch_df['question_text'].tolist()
        question_ids = batch_df['question_id'].tolist()
        
        responses = generator.generate_responses_batch(
            questions, 
            args.num_responses, 
            args.temperature,
            args.max_new_tokens
        )
        
        # Organize responses
        response_idx = 0
        for q_idx, (question_id, question) in enumerate(zip(question_ids, questions)):
            for resp_num in range(args.num_responses):
                all_responses_data.append({
                    'question_id': question_id,
                    'question_text': question,
                    'response_id': resp_num + 1,
                    'response_text': responses[response_idx]
                })
                response_idx += 1
        
        # Save checkpoint every 3 batches
        if (batch_idx + 1) % 3 == 0:
            checkpoint_file = f'{gpu_output_dir}/checkpoint_{batch_idx + 1}.csv'
            pd.DataFrame(all_responses_data).to_csv(checkpoint_file, index=False)
            print(f"[GPU {args.gpu_id}] Checkpoint saved")
    
    # Save final results
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    final_output = f'{gpu_output_dir}/{input_basename}_responses.csv'
    final_df = pd.DataFrame(all_responses_data)
    final_df.to_csv(final_output, index=False)
    
    # Clean up checkpoints
    for f in glob.glob(f"{gpu_output_dir}/checkpoint_*.csv"):
        os.remove(f)
    
    # Summary
    print(f"\n[GPU {args.gpu_id}] Complete!")
    print(f"[GPU {args.gpu_id}] Total responses: {len(all_responses_data)}")
    print(f"[GPU {args.gpu_id}] Unique questions: {final_df['question_id'].nunique()}")
    print(f"[GPU {args.gpu_id}] Saved to: {final_output}")


if __name__ == "__main__":
    main()
