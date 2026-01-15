"""
Therapeutic Response Generator

Generates multiple therapeutic responses per question using Mistral-7B-Instruct-v0.2.
This is the standard single-GPU version for most users.

For multi-GPU parallel processing, see `generate_responses_parallel.py`.

Usage:
    python generate_responses.py \
        --input_path ./processed/questions_train.csv \
        --output_path ./processed/responses_train.csv \
        --num_responses 5
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate therapeutic responses using Mistral-7B-Instruct"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV with questions (question_id, question_text)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated responses"
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses to generate per question (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save checkpoint every N questions (default: 100)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model name"
    )
    return parser.parse_args()


class TherapyResponseGenerator:
    """Generator for therapeutic responses using instruction-tuned LLMs."""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def create_prompt(self, question: str) -> str:
        """Create the instruction prompt for therapeutic response."""
        return f"[INST] You are a therapist. Respond to this person's question:\n{question} [/INST]"
    
    def generate_responses(
        self,
        questions: list[str],
        num_responses: int = 5,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        batch_size: int = 8
    ) -> list[list[str]]:
        """
        Generate multiple responses for each question.
        
        Args:
            questions: List of therapeutic questions
            num_responses: Number of responses per question
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation
            
        Returns:
            List of lists, where each inner list contains responses for one question
        """
        all_responses = []
        
        for question in tqdm(questions, desc="Generating responses"):
            question_responses = []
            prompt = self.create_prompt(question)
            
            # Generate responses in batches
            for i in range(0, num_responses, batch_size):
                current_batch_size = min(batch_size, num_responses - i)
                
                # Tokenize
                inputs = self.tokenizer(
                    [prompt] * current_batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.model.device)
                
                # Generate
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
                
                # Decode responses
                for output in outputs:
                    generated_tokens = output[inputs.input_ids.shape[1]:]
                    response = self.tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    ).strip()
                    question_responses.append(response)
                
                # Clear GPU memory
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_responses.append(question_responses)
        
        return all_responses


def load_checkpoint(checkpoint_path: Path) -> tuple[pd.DataFrame, set]:
    """Load existing checkpoint and return processed question IDs."""
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        processed_ids = set(df['question_id'].unique())
        print(f"Loaded checkpoint with {len(processed_ids)} processed questions")
        return df, processed_ids
    return None, set()


def save_checkpoint(data: list[dict], checkpoint_path: Path):
    """Save checkpoint to disk."""
    df = pd.DataFrame(data)
    df.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved: {len(df)} responses")


def main():
    args = parse_args()
    
    # Setup paths
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.csv"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Therapeutic Response Generation")
    print("=" * 60)
    
    # Load input data
    print(f"\nLoading questions from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} questions")
    
    # Handle resume
    all_responses_data = []
    processed_ids = set()
    
    if args.resume:
        existing_df, processed_ids = load_checkpoint(checkpoint_path)
        if existing_df is not None:
            all_responses_data = existing_df.to_dict('records')
            df = df[~df['question_id'].isin(processed_ids)]
            print(f"Remaining questions to process: {len(df)}")
    
    if len(df) == 0:
        print("All questions already processed!")
        return
    
    # Initialize generator
    generator = TherapyResponseGenerator(args.model_name)
    
    # Process questions
    questions = df['question_text'].tolist()
    question_ids = df['question_id'].tolist()
    
    print(f"\nGenerating {args.num_responses} responses per question...")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    
    for idx, (question_id, question) in enumerate(tqdm(
        zip(question_ids, questions), 
        total=len(questions),
        desc="Processing questions"
    )):
        # Generate responses for this question
        responses = generator.generate_responses(
            [question],
            num_responses=args.num_responses,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )[0]
        
        # Store responses
        for resp_num, response in enumerate(responses, 1):
            all_responses_data.append({
                'question_id': question_id,
                'question_text': question,
                'response_id': resp_num,
                'response_text': response
            })
        
        # Save checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(all_responses_data, checkpoint_path)
    
    # Save final output
    final_df = pd.DataFrame(all_responses_data)
    final_df.to_csv(output_path, index=False)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total questions: {final_df['question_id'].nunique()}")
    print(f"Total responses: {len(final_df)}")
    print(f"Responses per question: {args.num_responses}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
