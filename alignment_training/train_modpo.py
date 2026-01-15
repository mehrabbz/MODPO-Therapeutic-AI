"""
Multi-Objective Direct Preference Optimization (MODPO) for Therapeutic AI

MODPO extends DPO to support multiple objectives by incorporating margin rewards
from pre-trained reward models directly into the loss function.

Usage:
    python train_modpo.py \
        --preferences ./preferences/empathy_pairs.json \
        --reward_models safety:./reward_models/safety/final_model \
        --output_dir ./models/modpo_empathy \
        --primary_criterion empathy \
        --weights 0.5 0.5

Reference: Zhou et al., "Beyond One-Preference-Fits-All Alignment: Multi-Objective 
Direct Preference Optimization" (ACL 2024)
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
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import DPOConfig

from trainers import MODPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MODPO model")
    parser.add_argument("--preferences", type=str, required=True,
                        help="Path to primary preference pairs")
    parser.add_argument("--reward_models", type=str, nargs='+', required=True,
                        help="Margin reward models as name:path pairs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--primary_criterion", type=str, required=True,
                        help="Primary criterion name")
    parser.add_argument("--weights", type=float, nargs='+', default=None,
                        help="Weights for objectives [primary, margin1, ...]. Default: equal")
    parser.add_argument("--model_name", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_hpo", action="store_true")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def clear_cuda_memory():
    """Clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def load_preference_data(file_path: str) -> list[dict]:
    """Load preference data from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def parse_reward_model_args(reward_model_args: list[str]) -> dict[str, str]:
    """Parse reward model arguments into name:path dict."""
    reward_models = {}
    for arg in reward_model_args:
        name, path = arg.split(":")
        reward_models[name] = path
    return reward_models


def load_reward_models(reward_model_paths: dict[str, str], device: str):
    """Load reward models for margin computation."""
    reward_models = {}
    reward_tokenizers = {}
    
    for name, path in reward_model_paths.items():
        logger.info(f"Loading reward model: {name} from {path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        reward_models[name] = model
        reward_tokenizers[name] = tokenizer
    
    return reward_models, reward_tokenizers


def compute_margin_values(
    data: list[dict],
    reward_models: dict,
    reward_tokenizers: dict,
    weights: list[float],
    device: str,
) -> list[float]:
    """
    Compute margin values for each preference pair.
    
    margin = Σ_{j≠k} w_j * (r_j(x, y_w) - r_j(x, y_l))
    """
    margin_values = []
    
    for item in data:
        question = item["question_text"]
        chosen = item["chosen"]["response_text"]
        rejected = item["rejected"]["response_text"]
        
        margin = 0.0
        
        for idx, (name, model) in enumerate(reward_models.items()):
            tokenizer = reward_tokenizers[name]
            weight = weights[idx + 1]  # Skip primary weight (index 0)
            
            # Score chosen
            chosen_input = f"Question: {question}\nResponse: {chosen}"
            chosen_tokens = tokenizer(
                chosen_input, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to(device)
            
            with torch.no_grad():
                chosen_score = model(**chosen_tokens).logits.item()
            
            # Score rejected
            rejected_input = f"Question: {question}\nResponse: {rejected}"
            rejected_tokens = tokenizer(
                rejected_input, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to(device)
            
            with torch.no_grad():
                rejected_score = model(**rejected_tokens).logits.item()
            
            # Add weighted margin
            margin += weight * (chosen_score - rejected_score)
        
        margin_values.append(margin)
    
    return margin_values


def format_for_modpo(example: dict, tokenizer) -> dict:
    """Format example for MODPO training."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question_text"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    return {
        "prompt": prompt,
        "chosen": example["chosen"]["response_text"],
        "rejected": example["rejected"]["response_text"],
        "margin_values": example["margin_values"],
    }


def create_model_and_tokenizer(model_name: str, gpu_id: int):
    """Create model with quantization and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def train_modpo(
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    model_name: str,
    weights: list[float],
    num_objectives: int,
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    num_epochs: int = 3,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 4,
    gpu_id: int = 0,
    save_model: bool = True,
):
    """Train MODPO model."""
    clear_cuda_memory()
    
    model, _ = create_model_and_tokenizer(model_name, gpu_id)
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
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
    
    trainer = MODPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        weights=weights,
        num_objectives=num_objectives,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    trainer.train()
    
    if save_model:
        final_path = Path(output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Model saved to {final_path}")
    
    eval_results = trainer.evaluate()
    
    del model, trainer
    clear_cuda_memory()
    
    return str(final_path) if save_model else None, eval_results


def run_hpo(train_dataset, eval_dataset, tokenizer, model_name, weights, 
            num_objectives, gpu_id, n_trials, random_state):
    """Run hyperparameter optimization."""
    
    def objective(trial):
        clear_cuda_memory()
        
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "num_epochs": trial.suggest_int("num_epochs", 1, 5),
            "beta": trial.suggest_float("beta", 0.05, 0.5),
            "lora_r": trial.suggest_categorical("lora_r", [32, 64, 128]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [64, 128, 256]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.1),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [2, 4, 8]
            ),
        }
        
        try:
            _, eval_results = train_modpo(
                train_dataset, eval_dataset, tokenizer,
                output_dir=f"./hpo_trial_{trial.number}",
                model_name=model_name,
                weights=weights,
                num_objectives=num_objectives,
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
    
    return study.best_params


def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    
    logger.info(f"Training MODPO model for primary criterion: {args.primary_criterion}")
    
    # Parse reward model arguments
    reward_model_paths = parse_reward_model_args(args.reward_models)
    num_objectives = 1 + len(reward_model_paths)  # Primary + margin objectives
    
    # Set weights
    if args.weights is None:
        weights = [1.0 / num_objectives] * num_objectives
    else:
        weights = args.weights
    
    logger.info(f"Objectives: {args.primary_criterion} + {list(reward_model_paths.keys())}")
    logger.info(f"Weights: {weights}")
    
    # Load data
    preference_data = load_preference_data(args.preferences)
    logger.info(f"Loaded {len(preference_data)} preference pairs")
    
    # Load reward models and compute margins
    logger.info("Loading reward models for margin computation...")
    reward_models, reward_tokenizers = load_reward_models(reward_model_paths, device)
    
    logger.info("Computing margin values...")
    margin_values = compute_margin_values(
        preference_data, reward_models, reward_tokenizers, weights, device
    )
    
    # Add margin values to data
    for item, margin in zip(preference_data, margin_values):
        item["margin_values"] = margin
    
    # Cleanup reward models
    del reward_models, reward_tokenizers
    clear_cuda_memory()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format data
    dataset = Dataset.from_list(preference_data)
    dataset = dataset.map(lambda x: format_for_modpo(x, tokenizer))
    
    # Split
    split = dataset.train_test_split(test_size=0.1, seed=args.random_state)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Get hyperparameters
    if args.skip_hpo:
        best_params = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "beta": args.beta,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_accumulation_steps": 4,
        }
    else:
        logger.info(f"Running HPO with {args.n_trials} trials...")
        best_params = run_hpo(
            train_dataset, eval_dataset, tokenizer, args.model_name,
            weights, num_objectives, args.gpu_id, args.n_trials, args.random_state
        )
    
    # Final training
    logger.info("Training final model...")
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = output_dir / f"modpo_{args.primary_criterion}_{timestamp}"
    
    model_path, eval_results = train_modpo(
        train_dataset, eval_dataset, tokenizer,
        output_dir=str(final_output),
        model_name=args.model_name,
        weights=weights,
        num_objectives=num_objectives,
        gpu_id=args.gpu_id,
        save_model=True,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        **best_params
    )
    
    # Save info
    with open(final_output / "training_info.json", 'w') as f:
        json.dump({
            "primary_criterion": args.primary_criterion,
            "margin_objectives": list(reward_model_paths.keys()),
            "weights": weights,
            "model_name": args.model_name,
            "best_params": best_params,
            "eval_results": eval_results,
            "timestamp": timestamp
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("MODPO Training Complete!")
    print("=" * 60)
    print(f"Primary: {args.primary_criterion}")
    print(f"Margin objectives: {list(reward_model_paths.keys())}")
    print(f"Model saved to: {final_output}")


if __name__ == "__main__":
    main()
