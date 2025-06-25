#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
A comprehensive script to run a specific activation patching experiment across
a user-defined set of transformer models.

This script uses the experimental logic from the user-provided notebook and
integrates the model-loading framework and model list from the user-provided
script.

Key Features:
- Runs experiments on the user-specified list of models (GPT-2, GPT-J, Llama, etc.).
- Uses the original single-prompt ("toy data") experimental design.
- Dynamically determines model configuration and hook names for patching.
- Saves all numerical results (CSVs) and plots (PDFs) into a structured
  'Results/{model_name}/' directory.
- Includes an explicit variable for a Hugging Face token for loading gated models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Dict, Callable, List

# --- 1. ADVANCED MODEL SETUP (Inspired by user-provided script) ---

def get_model_config(model: nn.Module, model_name: str) -> Dict:
    """
    Creates a standardized configuration dictionary for various model architectures.
    This provides the correct hook names needed for our specific patching experiment.
    """
    config = {}
    model_name_lower = model_name.lower()

    if 'gpt-j' in model_name_lower:
        config.update({
            "n_layers": model.config.n_layer, "n_heads": model.config.n_head,
            "d_model": model.config.n_embd,
            "attn_hook_name_template": "transformer.h.{}.attn.out_proj",
            "mlp_hook_name_template": "transformer.h.{}.mlp.fc_out"
        })
    elif 'gpt2' in model_name_lower:
        config.update({
            "n_layers": model.config.n_layer, "n_heads": model.config.n_head,
            "d_model": model.config.n_embd,
            "attn_hook_name_template": "transformer.h.{}.attn.c_proj",
            "mlp_hook_name_template": "transformer.h.{}.mlp.c_proj"
        })
    elif 'gpt-neo' in model_name_lower and 'gpt-neox' not in model_name_lower:
        config.update({
            "n_layers": model.config.num_layers, "n_heads": model.config.num_heads,
            "d_model": model.config.hidden_size,
            "attn_hook_name_template": "transformer.h.{}.attn.out_proj",
            "mlp_hook_name_template": "transformer.h.{}.mlp.c_proj"
        })
    elif 'gpt-neox' in model_name_lower or 'pythia' in model_name_lower:
        config.update({
            "n_layers": model.config.num_hidden_layers, "n_heads": model.config.num_attention_heads,
            "d_model": model.config.hidden_size,
            "attn_hook_name_template": "gpt_neox.layers.{}.attention.dense",
            "mlp_hook_name_template": "gpt_neox.layers.{}.mlp.dense_4h_to_h"
        })
    elif 'llama' in model_name_lower:
        config.update({
            "n_layers": model.config.num_hidden_layers, "n_heads": model.config.num_attention_heads,
            "d_model": model.config.hidden_size,
            "attn_hook_name_template": "model.layers.{}.self_attn.o_proj",
            "mlp_hook_name_template": "model.layers.{}.mlp.down_proj"
        })
    else:
        raise NotImplementedError(f"Model architecture for '{model_name}' not recognized. Please add its configuration.")

    return config

def setup_model_and_tokenizer(model_name: str, device: str = 'cuda'):
    """
    Loads a pretrained Hugging Face model and tokenizer, handling various architectures.
    """
    print(f"--- Loading model and tokenizer for '{model_name}' ---")

    # --- ADD HUGGING FACE TOKEN HERE FOR GATED MODELS LIKE LLAMA ---
    # Replace "YOUR_HF_TOKEN_HERE" with your actual token.
    # It can be a read-only token for security.
    HUGGING_FACE_TOKEN = "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"

    model_dtype = torch.float16 if any(k in model_name.lower() for k in ['6b', '13b', '20b', '70b']) else torch.float32

    if 'llama' in model_name.lower():
        if HUGGING_FACE_TOKEN == "YOUR_HF_TOKEN_HERE":
            print("Warning: Llama model selected, but no Hugging Face token provided. This may fail.")
            access_token = None
        else:
            access_token = HUGGING_FACE_TOKEN

        tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, low_cpu_mem_usage=True, token=access_token).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, low_cpu_mem_usage=True).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    model_config = get_model_config(model, model_name)

    return model, tokenizer, model_config


# --- 2. DATASET DEFINITION (From user's notebook) ---
def get_task_datasets() -> Dict[str, Dict]:
    """
    Defines the clean and corrupted prompts and answers for each task.
    This uses the single-prompt ("toy data") setup as specified.
    """
    datasets = {
        "analogy": {
            "description": "Letter-String Analogy Task ('+1' vs No Rule)",
            "clean_prompt": "ABCD:ABCE::JKLM:JKL",
            "clean_correct_answer": "N",
            "clean_incorrect_answer": "M",
            "corrupted_prompt": "ABCD:ABCD::JKLM:JKL",
            "corrupted_correct_answer": "M",
            "corrupted_incorrect_answer": "N",
        },
        "sequencing": {
            "description": "Next-Item Sequencing Task",
            "clean_prompt": "January:February::Wednesday:",
            "clean_correct_answer": "Thursday",
            "clean_incorrect_answer": "Wednesday",
            "corrupted_prompt": "January:January::Wednesday:",
            "corrupted_correct_answer": "Wednesday",
            "corrupted_incorrect_answer": "Thursday",
        }
    }
    return datasets


# --- 3. CORE ACTIVATION PATCHING LOGIC (From user's notebook) ---
activation_cache: Dict[str, torch.Tensor] = {}

def caching_hook_factory(hook_name: str) -> Callable:
    def hook(module, input, output):
        tensor_to_cache = output[0] if isinstance(output, tuple) else output
        activation_cache[hook_name] = tensor_to_cache.detach()
    return hook

def patching_hook_factory(hook_name: str, head_index: int = None, d_head: int = None) -> Callable:
    def hook(module, input, output):
        if hook_name not in activation_cache: raise ValueError(f"Activation for {hook_name} not found!")
        cached_activation = activation_cache[hook_name]
        patched_output = output.clone()
        min_seq_len = min(patched_output.shape[-2], cached_activation.shape[-2])
        if head_index is not None:
            start, end = head_index * d_head, (head_index + 1) * d_head
            if patched_output.ndim == 3: patched_output[:, :min_seq_len, start:end] = cached_activation[:, :min_seq_len, start:end]
            elif patched_output.ndim == 2: patched_output[:min_seq_len, start:end] = cached_activation[:min_seq_len, start:end]
        else:
            if patched_output.ndim == 3: patched_output[:, :min_seq_len, :] = cached_activation[:, :min_seq_len, :]
            elif patched_output.ndim == 2: patched_output[:min_seq_len, :] = cached_activation[:min_seq_len, :]
        return patched_output
    return hook

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for part in name.split('.'): model = getattr(model, part)
    return model

def run_with_hooks(model: nn.Module, tokenizer: GPT2Tokenizer, prompt: str, hook_fns: Dict[str, Callable]) -> torch.Tensor:
    handles = []
    try:
        for name, hook_fn in hook_fns.items():
            module = get_module_by_name(model, name)
            handles.append(module.register_forward_hook(hook_fn))
        inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad(): outputs = model(**inputs)
        return outputs.logits[0, -1, :]
    finally:
        for handle in handles: handle.remove()

def calculate_logit_diff(logits: torch.Tensor, tokenizer: GPT2Tokenizer, correct_answer: str, incorrect_answer: str) -> float:
    try:
        correct_id = tokenizer.encode(correct_answer, add_prefix_space=False)[0]
        incorrect_id = tokenizer.encode(incorrect_answer, add_prefix_space=False)[0]
        return (logits[correct_id] - logits[incorrect_id]).item()
    except IndexError: return 0.0

# --- 4. EXPERIMENT EXECUTION (ADAPTED FOR MODEL_CONFIG) ---
def perform_patching_experiment(model: nn.Module, tokenizer: GPT2Tokenizer, model_config: Dict, source_prompt: str, dest_prompt: str, dest_correct_answer: str, dest_incorrect_answer: str, layer: int, component_type: str, head_index: int = None) -> float:
    global activation_cache
    activation_cache = {}
    hook_template = model_config['mlp_hook_name_template'] if component_type == 'mlp' else model_config['attn_hook_name_template']
    hook_name = hook_template.format(layer)
    run_with_hooks(model, tokenizer, source_prompt, {hook_name: caching_hook_factory(hook_name)})
    d_head = model_config["d_model"] // model_config["n_heads"] if component_type == 'attn_head' else None
    patching_hooks = {hook_name: patching_hook_factory(hook_name, head_index, d_head)}
    patched_logits = run_with_hooks(model, tokenizer, dest_prompt, patching_hooks)
    return calculate_logit_diff(patched_logits, tokenizer, dest_correct_answer, dest_incorrect_answer)

def run_exploratory_sweep(model: nn.Module, tokenizer: GPT2Tokenizer, model_config: Dict, task_data: Dict, patch_type: str) -> pd.DataFrame:
    n_layers, n_heads = model_config["n_layers"], model_config["n_heads"]
    if patch_type == 'noising':
        source_prompt, dest_prompt = task_data['corrupted_prompt'], task_data['clean_prompt']
        dest_correct, dest_incorrect = task_data['clean_correct_answer'], task_data['clean_incorrect_answer']
    else: # denoising
        source_prompt, dest_prompt = task_data['clean_prompt'], task_data['corrupted_prompt']
        dest_correct, dest_incorrect = task_data['corrupted_correct_answer'], task_data['corrupted_incorrect_answer']

    clean_logits = run_with_hooks(model, tokenizer, task_data['clean_prompt'], {})
    clean_baseline = calculate_logit_diff(clean_logits, tokenizer, task_data['clean_correct_answer'], task_data['clean_incorrect_answer'])
    corrupted_logits = run_with_hooks(model, tokenizer, task_data['corrupted_prompt'], {})
    corrupted_baseline = calculate_logit_diff(corrupted_logits, tokenizer, task_data['corrupted_correct_answer'], task_data['corrupted_incorrect_answer'])

    print(f"\nRunning {patch_type} sweep for '{task_data['description']}'")

    results = []
    for component_type, head_range in [('attn_head', range(n_heads)), ('mlp', [-1])]:
        print(f"  - Patching {component_type}s...")
        for layer in range(n_layers):
            for head_index in head_range:
                patched_logit_diff = perform_patching_experiment(model, tokenizer, model_config, source_prompt, dest_prompt, dest_correct, dest_incorrect, layer, component_type, head_index if component_type == 'attn_head' else None)
                if patch_type == 'noising':
                    effect = patched_logit_diff - clean_baseline
                else:
                    effect = patched_logit_diff - corrupted_baseline
                results.append({'layer': layer, 'head': head_index, 'type': component_type, 'effect': effect})

    return pd.DataFrame(results)

# --- 5. VISUALIZATION ---
def plot_results(df: pd.DataFrame, title: str, component_type: str, output_path: str = None):
    if component_type == 'attn_head':
        if df[df['type'] == 'attn_head'].empty: return
        pivot_df = df[df['type'] == 'attn_head'].pivot(index='head', columns='layer', values='effect')
        fig, ax = plt.subplots(figsize=(12, 10))
        max_abs_val = pivot_df.abs().max().max() if not pivot_df.empty else 1.0
        im = ax.imshow(pivot_df, cmap='coolwarm', vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax); cbar.ax.set_ylabel("Effect on Logit Difference", rotation=-90, va="bottom")
        ax.set_xticks(np.arange(pivot_df.shape[1])); ax.set_yticks(np.arange(pivot_df.shape[0]))
        ax.set_xticklabels(pivot_df.columns); ax.set_yticklabels(pivot_df.index)
        ax.set_xlabel("Layer"); ax.set_ylabel("Head Index")
    elif component_type == 'mlp':
        if df[df['type'] == 'mlp'].empty: return
        mlp_df = df[df['type'] == 'mlp'].sort_values('layer')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(mlp_df['layer'], mlp_df['effect'], color='skyblue')
        ax.set_xlabel("Layer"); ax.set_ylabel("Average Effect on Logit Difference")
        ax.grid(axis='y', linestyle='--'); ax.set_xticks(mlp_df['layer'])
    else: raise ValueError("Invalid component type")
    ax.set_title(title); fig.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf'); print(f"Saved plot to: {output_path}")
    plt.show(); plt.close(fig)

# --- 6. MAIN EXECUTION BLOCK ---

def main():
    """
    Main function to run the activation patching experiment across multiple models.
    """
    # This model dictionary is taken directly from the user's script
    models_to_test = {
        #'gptneo': 'EleutherAI/gpt-neo-125m',
        #'gpt2': 'gpt2', # Added gpt2 for a quick baseline
        'gptj6b': 'EleutherAI/gpt-j-6B',
        'llama27b': 'meta-llama/Llama-2-7b-hf',
        'llama213b': 'meta-llama/Llama-2-13b-hf',
        'gptneox20b': 'EleutherAI/gpt-neox-20b',
        'llama270b': 'meta-llama/Llama-2-70b-hf'
    }

    main_output_dir = "Results"
    os.makedirs(main_output_dir, exist_ok=True)
    datasets = get_task_datasets()

    for model_short_name, model_hf_name in models_to_test.items():
        try:
            model, tokenizer, model_config = setup_model_and_tokenizer(model_hf_name)
        except Exception as e:
            print(f"\n--- Could not load model {model_hf_name}. Skipping. Error: {e} ---\n")
            continue

        # Create model-specific subdirectory inside the main "Results" folder
        model_results_dir = os.path.join(main_output_dir, model_short_name)
        os.makedirs(model_results_dir, exist_ok=True)

        for task_name, task_data in datasets.items():
            for patch_type in ['noising', 'denoising']:
                result_key = f"{task_name}_{patch_type}"
                df = run_exploratory_sweep(model, tokenizer, model_config, task_data, patch_type)

                # Save CSV and Plots inside the model-specific subfolder
                csv_path = os.path.join(model_results_dir, f"{result_key}_results.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved data to: {csv_path}")

                plot_path_attn = os.path.join(model_results_dir, f"{result_key}_attn_heads.pdf")
                plot_path_mlp = os.path.join(model_results_dir, f"{result_key}_mlp_layers.pdf")
                title_attn = f"Attention Heads Effect ({patch_type.capitalize()})\n{model_short_name} - {datasets[task_name]['description']}"
                title_mlp = f"MLP Layers Effect ({patch_type.capitalize()})\n{model_short_name} - {datasets[task_name]['description']}"
                plot_results(df, title_attn, 'attn_head', output_path=plot_path_attn)
                plot_results(df, title_mlp, 'mlp', output_path=plot_path_mlp)

        print(f"--- Finished with {model_short_name}. Clearing memory. ---")
        del model, tokenizer, model_config
        torch.cuda.empty_cache()

        # --- NEW: Delete ONLY this model's cache ---
        from transformers import file_utils
        import shutil
        import re

        # 1. Get model's cache folder name (convert "/" to "--")
        model_cache_name = f"models--{re.sub(r'/', '--', model_hf_name)}"
        cache_path = os.path.join(file_utils.default_cache_path, model_cache_name)

        # 2. Delete only this model's folder
        if os.path.exists(cache_path):
            print(f"Deleting model cache: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)

if __name__ == '__main__':
    main()

