"""
Joint-Loss DPO Training for Therapeutic AI

A multi-objective DPO variant that combines multiple preference objectives into
a single sigmoid-weighted loss function. Unlike MODPO which uses margin rewards,
Joint-Loss DPO treats all criteria as primary objectives with separate preference
pairs for each.

Usage:
    python train_joint_loss_dpo.py \
        --preferences empathy:./preferences/empathy_pairs.json \
                      safety:./preferences/safety_pairs.json \
        --output_dir ./models/joint_loss_dpo \
        --weights 0.5 0.5
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
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import DPOConfig

from trainers import JointLossDPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Joint-Loss DPO model")
    parser.add_argument("--preferences", type=str, nargs='+', required=True,
                        help="Preference files as name:path pairs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--weights", type=float, nargs='+', default=None,
                        help="Weights for objectives. Default: equal")
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


def parse_preference_args(pref_args: list[str]) -> dict[str, str]:
    """Parse preference arguments into name:path dict."""
    prefs = {}
    for arg in pref_args:
        name, path = arg.split(":")
        prefs[name] = path
    return prefs


def load_all_preferences(pref_paths: dict[str, str]) -> dict[str, list[dict]]:
    """Load all preference files."""
    all_prefs = {}
    for name, path in pref_paths.items():
        with open(path, 'r') as f:
            all_prefs[name] = json.load(f)
        logger.info(f"Loaded {len(all_prefs[name])} pairs for {name}")
    return all_prefs


def align_preference_data(all_prefs: dict[str, list[dict]]) -> list[dict]:
    """
    Align preference data across objectives by question_id.
    
    Returns list of dicts with all objectives' chosen/rejected for each question.
    """
    # Get common question IDs
    question_ids_per_obj = [
        set(item["question_id"] for item in prefs)
        for prefs in all_prefs.values()
    ]
    common_ids = set.intersection(*question_ids_per_obj)
    logger.info(f"Common question IDs across all objectives: {len(common_ids)}")
    
    # Build lookup dicts
    lookups = {}
    for name, prefs in all_prefs.items():
        lookups[name] = {item["question_id"]: item for item in prefs}
    
    # Align data
    aligned_data = []
    obj_names = list(all_prefs.keys())
    
    for qid in common_ids:
        aligned_item = {
            "question_id": qid,
            "question_text": lookups[obj_names[0]][qid]["question_text"],
        }
        
        for idx, name in enumerate(obj_names):
            item = lookups[name][qid]
            aligned_item[f"chosen_{idx}"] = item["chosen"]["response_text"]
            aligned_item[f"rejected_{idx}"] = item["rejected"]["response_text"]
        
        aligned_data.append(aligned_item)
    
    return aligned_data, obj_names


def format_and_tokenize_multi_objective(
    example: dict,
    tokenizer,
    num_objectives: int,
) -> dict:
    """Format and tokenize for multi-objective training."""
    # Create prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question_text"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    result = {"prompt": prompt}
    
    # Tokenize each objective's chosen/rejected
    for k in range(num_objectives):
        chosen_text = prompt + example[f"chosen_{k}"]
        rejected_text = prompt + example[f"rejected_{k}"]
        
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )
        
        result[f"chosen_input_ids_{k}"] = chosen_tokens["input_ids"].squeeze()
        result[f"chosen_attention_mask_{k}"] = chosen_tokens["attention_mask"].squeeze()
        result[f"rejected_input_ids_{k}"] = rejected_tokens["input_ids"].squeeze()
        result[f"rejected_attention_mask_{k}"] = rejected_tokens["attention_mask"].squeeze()
    
    return result


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


def train_joint_loss_dpo(
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    model_name: str,
    num_objectives: int,
    weights: list[float],
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    num_epochs: int = 3,
    beta: float = 0.1,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 4,
    gpu_id: int = 0,
    save_model: bool = True,
):
    """Train Joint-Loss DPO model."""
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
    
    trainer = JointLossDPOTrainer(
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


def run_hpo(train_dataset, eval_dataset, tokenizer, model_name, num_objectives,
            weights, gpu_id, n_trials, random_state):
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
            _, eval_results = train_joint_loss_dpo(
                train_dataset, eval_dataset, tokenizer,
                output_dir=f"./hpo_trial_{trial.number}",
                model_name=model_name,
                num_objectives=num_objectives,
                weights=weights,
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
    
    # Parse preference files
    preference_files = parse_preference_args(args.preferences)
    num_objectives = len(preference_files)
    
    logger.info(f"Training Joint-Loss DPO with {num_objectives} objectives")
    logger.info(f"Objectives: {list(preference_files.keys())}")
    
    # Set weights
    if args.weights is None:
        weights = [1.0 / num_objectives] * num_objectives
    else:
        weights = args.weights
    
    logger.info(f"Weights: {weights}")
    
    # Load and align data
    all_prefs = load_all_preferences(preference_files)
    aligned_data, obj_names = align_preference_data(all_prefs)
    logger.info(f"Aligned {len(aligned_data)} examples across objectives")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format data
    dataset = Dataset.from_list(aligned_data)
    dataset = dataset.map(
        lambda x: format_and_tokenize_multi_objective(x, tokenizer, num_objectives),
        remove_columns=dataset.column_names
    )
    
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
            num_objectives, weights, args.gpu_id, args.n_trials, args.random_state
        )
    
    # Final training
    logger.info("Training final model...")
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    obj_str = "_".join(obj_names)
    final_output = output_dir / f"joint_loss_dpo_{obj_str}_{timestamp}"
    
    model_path, eval_results = train_joint_loss_dpo(
        train_dataset, eval_dataset, tokenizer,
        output_dir=str(final_output),
        model_name=args.model_name,
        num_objectives=num_objectives,
        weights=weights,
        gpu_id=args.gpu_id,
        save_model=True,
        **best_params
    )
    
    # Save info
    with open(final_output / "training_info.json", 'w') as f:
        json.dump({
            "objectives": obj_names,
            "weights": weights,
            "model_name": args.model_name,
            "best_params": best_params,
            "eval_results": eval_results,
            "timestamp": timestamp
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Joint-Loss DPO Training Complete!")
    print("=" * 60)
    print(f"Objectives: {obj_names}")
    print(f"Model saved to: {final_output}")


if __name__ == "__main__":
    main()
