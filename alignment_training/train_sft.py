"""
Supervised Fine-Tuning (SFT) for Therapeutic AI

Trains a language model using supervised fine-tuning on preferred responses
selected based on a specific therapeutic criterion.

Usage:
    python train_sft.py \
        --preferences ./preferences/empathy_pairs.json \
        --output_dir ./models/sft_empathy \
        --criterion empathy
"""

import argparse
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import optuna
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFT model for therapeutic AI")
    parser.add_argument("--preferences", type=str, required=True,
                        help="Path to preference pairs JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--criterion", type=str, required=True,
                        help="Therapeutic criterion name (for logging)")
    parser.add_argument("--model_name", type=str, 
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model name")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--skip_hpo", action="store_true",
                        help="Skip hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of HPO trials")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def clear_cuda_memory():
    """Clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def load_preference_data(file_path: str) -> list[dict]:
    """Load preference data and extract chosen responses for SFT."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract only chosen responses for SFT
    sft_data = []
    for item in data:
        sft_data.append({
            "question_id": item["question_id"],
            "question_text": item["question_text"],
            "response_text": item["chosen"]["response_text"]
        })
    
    return sft_data


def format_for_sft(example: dict, tokenizer) -> dict:
    """Format example as instruction-response pair."""
    # Create chat format
    messages = [
        {"role": "user", "content": example["question_text"]},
        {"role": "assistant", "content": example["response_text"]}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return {"text": text}


def create_model_and_tokenizer(model_name: str, gpu_id: int):
    """Create model with quantization and tokenizer."""
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def train_sft(
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    model_name: str,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    num_epochs: int = 3,
    max_seq_length: int = 1024,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 4,
    gpu_id: int = 0,
    save_model: bool = True,
):
    """Train SFT model."""
    clear_cuda_memory()
    
    # Create model
    model, _ = create_model_and_tokenizer(model_name, gpu_id)
    
    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    trainer.train()
    
    # Save model
    if save_model:
        final_path = Path(output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Model saved to {final_path}")
    
    # Get final metrics
    eval_results = trainer.evaluate()
    
    # Cleanup
    del model, trainer
    clear_cuda_memory()
    
    return str(final_path) if save_model else None, eval_results


def run_hpo(train_dataset, eval_dataset, tokenizer, model_name, gpu_id, n_trials, random_state):
    """Run hyperparameter optimization using Optuna."""
    
    def objective(trial):
        clear_cuda_memory()
        
        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "num_epochs": trial.suggest_int("num_epochs", 1, 5),
            "lora_r": trial.suggest_categorical("lora_r", [32, 64, 128]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [64, 128, 256]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.1),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [2, 4, 8]
            ),
        }
        
        try:
            _, eval_results = train_sft(
                train_dataset, eval_dataset, tokenizer,
                output_dir=f"./hpo_trial_{trial.number}",
                model_name=model_name,
                gpu_id=gpu_id,
                save_model=False,
                **params
            )
            return eval_results["eval_loss"]
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")
        finally:
            clear_cuda_memory()
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return study.best_params


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    logger.info(f"Training SFT model for criterion: {args.criterion}")
    logger.info(f"Loading preferences from: {args.preferences}")
    
    # Load data
    sft_data = load_preference_data(args.preferences)
    logger.info(f"Loaded {len(sft_data)} training examples")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format data
    dataset = Dataset.from_list(sft_data)
    dataset = dataset.map(lambda x: format_for_sft(x, tokenizer))
    
    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=args.random_state)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Get hyperparameters
    if args.skip_hpo:
        best_params = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_accumulation_steps": 4,
        }
    else:
        logger.info(f"Running HPO with {args.n_trials} trials...")
        best_params = run_hpo(
            train_dataset, eval_dataset, tokenizer,
            args.model_name, args.gpu_id, args.n_trials, args.random_state
        )
    
    # Final training
    logger.info("Training final model...")
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = output_dir / f"sft_{args.criterion}_{timestamp}"
    
    model_path, eval_results = train_sft(
        train_dataset, eval_dataset, tokenizer,
        output_dir=str(final_output),
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        save_model=True,
        max_seq_length=args.max_seq_length,
        **best_params
    )
    
    # Save training info
    with open(final_output / "training_info.json", 'w') as f:
        json.dump({
            "criterion": args.criterion,
            "model_name": args.model_name,
            "best_params": best_params,
            "eval_results": eval_results,
            "timestamp": timestamp
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SFT Training Complete!")
    print("=" * 60)
    print(f"Criterion: {args.criterion}")
    print(f"Model saved to: {final_output}")
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
