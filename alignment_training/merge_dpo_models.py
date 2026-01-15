"""
DPO Soup: Parameter Merging for Multi-Objective Alignment

Merges parameters from separately trained DPO models (e.g., empathy and safety)
using linear interpolation to create a combined model that balances multiple
objectives without joint training.

Based on: Jang et al., "Personalized Soups: Personalized Large Language Model
Alignment via Post-hoc Parameter Merging" (arXiv:2310.11564)

Usage:
    # Merge two models with equal weights
    python merge_dpo_models.py \
        --models empathy:./models/dpo_empathy/final_model \
                 safety:./models/dpo_safety/final_model \
        --output_dir ./models/dpo_soup_empathy_safety
    
    # Merge with custom weights (must sum to 1.0)
    python merge_dpo_models.py \
        --models empathy:./models/dpo_empathy/final_model \
                 safety:./models/dpo_safety/final_model \
        --weights 0.6 0.4 \
        --output_dir ./models/dpo_soup_empathy_safety
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge DPO models using parameter averaging (DPO Soup)"
    )
    parser.add_argument(
        "--models", type=str, nargs='+', required=True,
        help="Models to merge as name:path pairs (e.g., empathy:./models/dpo_empathy)"
    )
    parser.add_argument(
        "--weights", type=float, nargs='+', default=None,
        help="Weights for each model (must sum to 1.0). Default: equal weights"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name (must match the base used for DPO training)"
    )
    parser.add_argument(
        "--merge_method", type=str, default="linear",
        choices=["linear", "slerp"],
        help="Merging method: 'linear' (weighted average) or 'slerp' (spherical interpolation)"
    )
    return parser.parse_args()


def parse_model_args(model_args: list[str]) -> dict[str, str]:
    """Parse model arguments into name:path dict."""
    models = {}
    for arg in model_args:
        parts = arg.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid model argument: {arg}. Expected format: name:path")
        name, path = parts
        models[name] = path
    return models


def load_lora_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter state dict from a saved model."""
    adapter_path = Path(model_path) / "adapter_model.safetensors"
    
    if adapter_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(adapter_path))
    else:
        # Try .bin format
        adapter_path = Path(model_path) / "adapter_model.bin"
        if adapter_path.exists():
            state_dict = torch.load(str(adapter_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No adapter found at {model_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin"
            )
    
    return state_dict


def linear_merge(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float]
) -> dict[str, torch.Tensor]:
    """
    Merge state dicts using weighted linear interpolation.
    
    merged_param = Σ_i (w_i * param_i)
    """
    merged = {}
    
    # Get all parameter names from first state dict
    param_names = list(state_dicts[0].keys())
    
    for name in param_names:
        merged_param = torch.zeros_like(state_dicts[0][name], dtype=torch.float32)
        
        for state_dict, weight in zip(state_dicts, weights):
            if name not in state_dict:
                raise KeyError(f"Parameter {name} not found in all models")
            merged_param += weight * state_dict[name].float()
        
        # Convert back to original dtype
        merged[name] = merged_param.to(state_dicts[0][name].dtype)
    
    return merged


def slerp_merge(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float]
) -> dict[str, torch.Tensor]:
    """
    Merge state dicts using spherical linear interpolation (SLERP).
    
    Only supports merging exactly 2 models.
    For vectors v1, v2 with interpolation factor t:
    slerp(v1, v2, t) = sin((1-t)θ)/sin(θ) * v1 + sin(tθ)/sin(θ) * v2
    where θ = arccos(v1·v2 / (|v1||v2|))
    """
    if len(state_dicts) != 2:
        raise ValueError("SLERP only supports merging exactly 2 models")
    
    merged = {}
    t = weights[1]  # Interpolation factor toward second model
    
    for name in state_dicts[0].keys():
        v1 = state_dicts[0][name].float().flatten()
        v2 = state_dicts[1][name].float().flatten()
        
        # Normalize
        v1_norm = torch.nn.functional.normalize(v1, dim=0)
        v2_norm = torch.nn.functional.normalize(v2, dim=0)
        
        # Compute angle
        dot = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
        theta = torch.acos(dot)
        
        # Handle edge cases
        if theta.abs() < 1e-6:
            # Vectors are nearly identical, use linear interpolation
            merged_flat = (1 - t) * v1 + t * v2
        elif (torch.pi - theta).abs() < 1e-6:
            # Vectors are nearly opposite, use linear interpolation
            merged_flat = (1 - t) * v1 + t * v2
        else:
            # Standard SLERP
            sin_theta = torch.sin(theta)
            merged_flat = (
                torch.sin((1 - t) * theta) / sin_theta * v1 +
                torch.sin(t * theta) / sin_theta * v2
            )
        
        # Reshape back
        merged[name] = merged_flat.reshape(state_dicts[0][name].shape).to(
            state_dicts[0][name].dtype
        )
    
    return merged


def save_merged_adapter(
    merged_state_dict: dict[str, torch.Tensor],
    source_model_path: str,
    output_path: str
):
    """Save merged adapter with config from source model."""
    import shutil
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy adapter config from source
    source_path = Path(source_model_path)
    config_file = source_path / "adapter_config.json"
    if config_file.exists():
        shutil.copy(config_file, output_path / "adapter_config.json")
    
    # Save merged weights
    from safetensors.torch import save_file
    save_file(merged_state_dict, str(output_path / "adapter_model.safetensors"))
    
    logger.info(f"Saved merged adapter to {output_path}")


def main():
    args = parse_args()
    
    # Parse model arguments
    model_paths = parse_model_args(args.models)
    model_names = list(model_paths.keys())
    num_models = len(model_paths)
    
    logger.info(f"Merging {num_models} models: {model_names}")
    
    # Validate and set weights
    if args.weights is None:
        weights = [1.0 / num_models] * num_models
    else:
        if len(args.weights) != num_models:
            raise ValueError(
                f"Number of weights ({len(args.weights)}) must match "
                f"number of models ({num_models})"
            )
        weights = args.weights
    
    # Validate weights sum to 1.0
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
        weights = [w / weight_sum for w in weights]
    
    logger.info(f"Using weights: {dict(zip(model_names, weights))}")
    
    # Load state dicts
    logger.info("Loading model adapters...")
    state_dicts = []
    for name, path in model_paths.items():
        logger.info(f"  Loading {name} from {path}")
        state_dict = load_lora_state_dict(path)
        state_dicts.append(state_dict)
        logger.info(f"    Loaded {len(state_dict)} parameters")
    
    # Verify all state dicts have same parameters
    param_names = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != param_names:
            missing = param_names - set(sd.keys())
            extra = set(sd.keys()) - param_names
            raise ValueError(
                f"Model {model_names[i]} has different parameters. "
                f"Missing: {missing}, Extra: {extra}"
            )
    
    # Merge
    logger.info(f"Merging with method: {args.merge_method}")
    if args.merge_method == "linear":
        merged_state_dict = linear_merge(state_dicts, weights)
    elif args.merge_method == "slerp":
        merged_state_dict = slerp_merge(state_dicts, weights)
    else:
        raise ValueError(f"Unknown merge method: {args.merge_method}")
    
    # Save
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_suffix = "_".join(model_names)
    final_output = output_dir / f"dpo_soup_{model_suffix}_{timestamp}"
    
    save_merged_adapter(
        merged_state_dict,
        source_model_path=list(model_paths.values())[0],  # Use first model's config
        output_path=str(final_output)
    )
    
    # Copy tokenizer from base model
    logger.info("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(str(final_output))
    
    # Save merge info
    merge_info = {
        "merged_models": model_paths,
        "weights": dict(zip(model_names, weights)),
        "merge_method": args.merge_method,
        "base_model": args.base_model,
        "timestamp": timestamp
    }
    
    with open(final_output / "merge_info.json", 'w') as f:
        json.dump(merge_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DPO Soup Merging Complete!")
    print("=" * 60)
    print(f"Merged models: {model_names}")
    print(f"Weights: {dict(zip(model_names, weights))}")
    print(f"Method: {args.merge_method}")
    print(f"Output: {final_output}")
    print("\nTo use the merged model:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.base_model}')")
    print(f"  model = PeftModel.from_pretrained(base, '{final_output}')")


if __name__ == "__main__":
    main()
