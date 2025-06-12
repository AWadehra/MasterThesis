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
A comprehensive, self-contained Python script to investigate and compare the
neural mechanisms of two distinct tasks in a transformer model using activation patching.

This script implements the methodology discussed in papers like "How to use and
interpret activation patching" (Heimersheim & Nanda, 2024) to identify which
attention heads and MLP layers are causally implicated in:
1.  Letter-String Analogy Task (e.g., ABCD:ABCE :: JKLM:JKL?) -> N
2.  Next-Item Sequencing Task (e.g., Monday, Tuesday, ...)

This definitive version (v4) is a complete research tool:
- It uses a highly precise experimental design with a "different rule" corrupted prompt.
- It calculates the final "effect" of each patch relative to the correct baseline.
- It saves all numerical results to CSV files for data persistence.
- It saves all generated plots as high-quality PDF files for reporting and also
  displays them for interactive viewing.

To run this script, you need to install the following libraries:
- pip install torch
- pip install transformers
- pip install matplotlib
- pip install seaborn
- pip install pandas
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Callable

# --- 1. SETUP: MODEL AND TOKENIZER ---

def setup_model_and_tokenizer(model_name: str = 'gpt2'):
    """
    Loads a pretrained GPT-2 model and its tokenizer.
    """
    print(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    return model, tokenizer

# --- 2. DATASET DEFINITION (FINAL) ---

def get_task_datasets() -> Dict[str, Dict]:
    """
    Defines the clean and corrupted prompts and answers for each task.
    """
    datasets = {
        "analogy": {
            "description": "Letter-String Analogy Task ('+1' vs 'Swap' Rule)",
            "clean_prompt": "ABCD:ABCE::JKLM:JKL",
            "clean_correct_answer": "N",
            "clean_incorrect_answer": "M",
            "corrupted_prompt": "ABCD:BACD::JKLM:KJL",
            "corrupted_correct_answer": "M",
            "corrupted_incorrect_answer": "N",
        },
        "sequencing": {
            "description": "Next-Item Sequencing Task",
            "clean_prompt": "January, February, March, April,",
            "clean_correct_answer": " May",
            "clean_incorrect_answer": " July",
            "corrupted_prompt": "January, Car, Plane, April,",
            "corrupted_correct_answer": " May",
            "corrupted_incorrect_answer": " July",
        }
    }
    return datasets


# --- 3. CORE ACTIVATION PATCHING LOGIC ---

activation_cache: Dict[str, torch.Tensor] = {}

def caching_hook_factory(hook_name: str) -> Callable:
    def hook(module, input, output):
        tensor_to_cache = output[0] if isinstance(output, tuple) else output
        activation_cache[hook_name] = tensor_to_cache.detach()
    return hook

def patching_hook_factory(hook_name: str, head_index: int = None, d_head: int = None) -> Callable:
    def hook(module, input, output):
        if hook_name not in activation_cache:
            raise ValueError(f"Activation for {hook_name} not found in cache!")
        cached_activation = activation_cache[hook_name]
        if head_index is not None:
            original_output = output[0]
            start, end = head_index * d_head, (head_index + 1) * d_head
            patched_output = original_output.clone()
            patched_output[:, :, start:end] = cached_activation[:, :, start:end]
            return (patched_output, output[1])
        else:
            return cached_activation
    return hook

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for part in name.split('.'):
        model = getattr(model, part)
    return model

def run_with_hooks(
        model: nn.Module,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        hook_fns: Dict[str, Callable],
) -> torch.Tensor:
    handles = []
    try:
        for name, hook_fn in hook_fns.items():
            module_name = name.split('_')[0] if 'attn.c_proj' in name else name
            module = get_module_by_name(model, module_name)
            handles.append(module.register_forward_hook(hook_fn))
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits[0, -1, :]
    finally:
        for handle in handles:
            handle.remove()

def calculate_logit_diff(
        logits: torch.Tensor,
        tokenizer: GPT2Tokenizer,
        correct_answer: str,
        incorrect_answer: str
) -> float:
    correct_id = tokenizer.encode(correct_answer, add_prefix_space=False)[0]
    incorrect_id = tokenizer.encode(incorrect_answer, add_prefix_space=False)[0]
    return (logits[correct_id] - logits[incorrect_id]).item()

# --- 4. EXPERIMENT EXECUTION ---

def perform_patching_experiment(
        model: nn.Module,
        tokenizer: GPT2Tokenizer,
        source_prompt: str,
        dest_prompt: str,
        dest_correct_answer: str,
        dest_incorrect_answer: str,
        layer: int,
        component_type: str,
        head_index: int = None
) -> float:
    global activation_cache
    activation_cache = {}

    if component_type == 'attn_head':
        hook_name = f"transformer.h.{layer}.attn.c_proj"
    elif component_type == 'mlp':
        hook_name = f"transformer.h.{layer}.mlp.c_proj"
    else:
        raise ValueError("Invalid component type")

    caching_hooks = {hook_name: caching_hook_factory(hook_name)}
    run_with_hooks(model, tokenizer, source_prompt, caching_hooks)

    d_head = model.config.n_embd // model.config.n_head if component_type == 'attn_head' else None
    patching_hooks = {hook_name: patching_hook_factory(hook_name, head_index, d_head)}
    patched_logits = run_with_hooks(model, tokenizer, dest_prompt, patching_hooks)

    return calculate_logit_diff(patched_logits, tokenizer, dest_correct_answer, dest_incorrect_answer)

def run_exploratory_sweep(
        model: nn.Module,
        tokenizer: GPT2Tokenizer,
        task_data: Dict,
        patch_type: str
) -> pd.DataFrame:
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    if patch_type == 'noising':
        source_prompt = task_data['corrupted_prompt']
        dest_prompt = task_data['clean_prompt']
        dest_correct_answer = task_data['clean_correct_answer']
        dest_incorrect_answer = task_data['clean_incorrect_answer']
    elif patch_type == 'denoising':
        source_prompt = task_data['clean_prompt']
        dest_prompt = task_data['corrupted_prompt']
        dest_correct_answer = task_data['corrupted_correct_answer']
        dest_incorrect_answer = task_data['corrupted_incorrect_answer']
    else:
        raise ValueError("patch_type must be 'noising' or 'denoising'")

    clean_logits = run_with_hooks(model, tokenizer, task_data['clean_prompt'], {})
    clean_baseline = calculate_logit_diff(clean_logits, tokenizer, task_data['clean_correct_answer'], task_data['clean_incorrect_answer'])

    corrupted_logits = run_with_hooks(model, tokenizer, task_data['corrupted_prompt'], {})
    corrupted_baseline = calculate_logit_diff(corrupted_logits, tokenizer, task_data['corrupted_correct_answer'], task_data['corrupted_incorrect_answer'])

    print(f"\nRunning {patch_type} sweep for '{task_data['description']}'")
    print(f"  - Clean Run Baseline Logit Diff: {clean_baseline:.4f}")
    print(f"  - Corrupted Run Baseline Logit Diff: {corrupted_baseline:.4f}")

    results = []
    print("  - Patching Attention Heads...")
    for layer in range(n_layers):
        for head in range(n_heads):
            patched_logit_diff = perform_patching_experiment(
                model, tokenizer, source_prompt, dest_prompt,
                dest_correct_answer, dest_incorrect_answer,
                layer, 'attn_head', head
            )
            results.append({'layer': layer, 'head': head, 'type': 'attn_head', 'patched_logit_diff': patched_logit_diff})

    print("  - Patching MLP Layers...")
    for layer in range(n_layers):
        patched_logit_diff = perform_patching_experiment(
            model, tokenizer, source_prompt, dest_prompt,
            dest_correct_answer, dest_incorrect_answer,
            layer, 'mlp'
        )
        results.append({'layer': layer, 'head': -1, 'type': 'mlp', 'patched_logit_diff': patched_logit_diff})

    df = pd.DataFrame(results)

    if patch_type == 'noising':
        df['effect'] = df['patched_logit_diff'] - clean_baseline
    else:
        df['effect'] = df['patched_logit_diff'] - corrupted_baseline

    return df

# --- 5. VISUALIZATION ---

def plot_heatmap(df: pd.DataFrame, title: str, component_type: str, output_path: str = None):
    """
    Plots a heatmap or bar chart and optionally saves it to a file.
    """
    if component_type == 'attn_head':
        pivot_df = df[df['type'] == 'attn_head'].pivot(index='layer', columns='head', values='effect')
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, cmap='coolwarm', center=0.0, annot=False)
        plt.xlabel("Head Index")
        plt.ylabel("Layer")
    elif component_type == 'mlp':
        mlp_df = df[df['type'] == 'mlp'].sort_values('layer')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='layer', y='effect', data=mlp_df, color='skyblue')
        plt.xlabel("Layer")
        plt.ylabel("Effect on Logit Difference")
        plt.grid(axis='y', linestyle='--')
    else:
        raise ValueError("Invalid component type for plotting")

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='pdf')
        print(f"Saved plot to: {output_path}")

    plt.show()
    plt.close()

# --- 6. MAIN EXECUTION ---

if __name__ == '__main__':
    MODEL_NAME = 'gpt2'
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    datasets = get_task_datasets()

    all_results = {}

    for task_name, task_data in datasets.items():
        for patch_type in ['noising', 'denoising']:
            result_key = f"{task_name}_{patch_type}"
            df = run_exploratory_sweep(model, tokenizer, task_data, patch_type)
            all_results[result_key] = df

    # --- Save Dataframes to CSV ---
    print("\n--- Saving Results to CSV ---")
    output_dir_data = "patching_results_data"
    if not os.path.exists(output_dir_data):
        os.makedirs(output_dir_data)

    for result_key, df in all_results.items():
        file_path = os.path.join(output_dir_data, f"{result_key}_results.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved data to: {file_path}")

    # --- Visualize and Save Plots to PDF ---
    print("\n--- Generating and Saving Plots ---")
    output_dir_plots = "patching_results_plots"
    if not os.path.exists(output_dir_plots):
        os.makedirs(output_dir_plots)

    for result_key, df in all_results.items():
        task_name, patch_type = result_key.split('_')

        plot_path_attn = os.path.join(output_dir_plots, f"{result_key}_attn_heads.pdf")
        plot_path_mlp = os.path.join(output_dir_plots, f"{result_key}_mlp_layers.pdf")

        title_attn = f"Attention Heads Effect ({patch_type.capitalize()})\n{datasets[task_name]['description']}"
        plot_heatmap(df, title_attn, 'attn_head', output_path=plot_path_attn)

        title_mlp = f"MLP Layers Effect ({patch_type.capitalize()})\n{datasets[task_name]['description']}"
        plot_heatmap(df, title_mlp, 'mlp', output_path=plot_path_mlp)

    print("\n--- End of Experiment ---")

