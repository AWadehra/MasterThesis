import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Dict, Callable, List
import json
import random
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===== CHANGE 1: ADD DETERMINISM FOUNDATION =====
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# ===== END CHANGE 1 =====

def get_model_config(model: nn.Module, model_name: str) -> Dict:
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
    print(f"--- Loading model and tokenizer for '{model_name}' ---")

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
    
    # ===== CHANGE 6: LAYERNORM STABILIZATION =====
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eps = 1e-3  # Increased from 1e-5/1e-6
    # ===== END CHANGE 6 =====
    
    model_config = get_model_config(model, model_name)

    return model, tokenizer, model_config

def load_gpt_model_and_tokenizer(model_name:str, device='cuda'):
    assert model_name is not None
    print(f"--- Loading model and tokenizer for '{model_name}' ---")

    HUGGING_FACE_TOKEN = "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"
    kwargs = {'low_cpu_mem_usage': True}
    
    if 'gpt-j' in model_name.lower():
        print("Using float16 revision for gpt-j-6B to ensure PyTorch-only workflow.")
        kwargs['revision'] = 'float16'
        kwargs['torch_dtype'] = torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    elif 'llama' in model_name.lower():
        access_token = HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN != "YOUR_HF_TOKEN_HERE" else None
        if not access_token: print("Warning: Llama model selected, but no Hugging Face token provided.")
        kwargs['token'] = access_token
        if '70b' in model_name.lower():
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', 
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            kwargs['quantization_config'] = bnb_config
            kwargs['trust_remote_code'] = True
            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token)
            model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            kwargs['torch_dtype'] = torch.float16 if any(k in model_name.lower() for k in ['13b']) else torch.float32
            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token)
            model = LlamaForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    else:
        kwargs['torch_dtype'] = torch.float16 if '20b' in model_name.lower() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    
    # ===== CHANGE 6: LAYERNORM STABILIZATION =====
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eps = 1e-3  # Increased from 1e-5/1e-6
    # ===== END CHANGE 6 =====
    
    model_config = {}
    model_name_lower = model_name.lower()
    if 'gpt-j' in model_name_lower:
        model_config.update({"n_layers": model.config.n_layer, "n_heads": model.config.n_head, "d_model": model.config.n_embd, "attn_hook_name_template": "transformer.h.{}.attn.out_proj", "mlp_hook_name_template": "transformer.h.{}.mlp.fc_out"})
    elif 'gpt2' in model_name_lower:
        model_config.update({"n_layers": model.config.n_layer, "n_heads": model.config.n_head, "d_model": model.config.n_embd, "attn_hook_name_template": "transformer.h.{}.attn.c_proj", "mlp_hook_name_template": "transformer.h.{}.mlp.c_proj"})
    elif 'gpt-neo' in model_name_lower and 'gpt-neox' not in model_name_lower:
        model_config.update({"n_layers": model.config.num_layers, "n_heads": model.config.num_heads, "d_model": model.config.hidden_size, "attn_hook_name_template": "transformer.h.{}.attn.out_proj", "mlp_hook_name_template": "transformer.h.{}.mlp.c_proj"})
    elif 'gpt-neox' in model_name_lower or 'pythia' in model_name_lower:
        model_config.update({"n_layers": model.config.num_hidden_layers, "n_heads": model.config.num_attention_heads, "d_model": model.config.hidden_size, "attn_hook_name_template": "gpt_neox.layers.{}.attention.dense", "mlp_hook_name_template": "gpt_neox.layers.{}.mlp.dense_4h_to_h"})
    elif 'llama' in model_name_lower:
        model_config.update({"n_layers": model.config.num_hidden_layers, "n_heads": model.config.num_attention_heads, "d_model": model.config.hidden_size, "attn_hook_name_template": "model.layers.{}.self_attn.o_proj", "mlp_hook_name_template": "model.layers.{}.mlp.down_proj"})
    else: raise NotImplementedError("Model architecture not recognized.")

    return model, tokenizer, model_config

def get_task_datasets() -> Dict[str, Dict]:
    # ===== CHANGE 2: SEED PROMPT SAMPLING =====
    random.seed(SEED)
    # ===== END CHANGE 2 =====
    
    with open('../BaselineAccuracy/dataset_files/abstractive/succ_letterstring_basic.json', 'r') as f:
        analogy_data = json.load(f)

    analogy_prompts = []
    for _ in range(20):
        # ===== CHANGE 2: USE CHOICE INSTEAD OF SAMPLE =====
        while True:
            example_pair, target_pair = random.sample(analogy_data, 2)
            if example_pair['input'] != target_pair['input']:
                break
        # ===== END CHANGE 2 =====

        example_in = example_pair['input'].strip('[]').replace(' ', '')
        example_out = example_pair['output'].strip('[]').replace(' ', '')
        target_in_full = target_pair['input'].strip('[]').replace(' ', '')
        target_out_full = target_pair['output'].strip('[]').replace(' ', '')

        # Create the prefix and single-token answers
        #target_in_parts = target_in_full.split(':')
        target_prefix = target_out_full[:3]
        correct_answer = target_out_full[-1]
        incorrect_answer = target_in_full[-1]

        analogy_prompts.append({
            "clean_prompt": f"{example_in}:{example_out}::{target_in_full}:{target_prefix}",
            "clean_correct_answer": correct_answer,
            "clean_incorrect_answer": incorrect_answer,
            "corrupted_prompt": f"{example_in}:{example_in}::{target_in_full}:{target_prefix}",
            "corrupted_correct_answer": incorrect_answer,
            "corrupted_incorrect_answer": correct_answer,
        })

    with open('../BaselineAccuracy/dataset_files/abstractive/next_item.json', 'r') as f:
        sequencing_data = json.load(f)

    sequencing_prompts = []
    for _ in range(20):
        # ===== CHANGE 2: USE CHOICE INSTEAD OF SAMPLE =====
        while True:
            example_pair, target_pair = random.sample(sequencing_data, 2)
            if example_pair['input'] != target_pair['input']:
                break
        # ===== END CHANGE 2 =====

        example_in, example_out = example_pair['input'], example_pair['output']
        target_in, target_out = target_pair['input'], target_pair['output']

        sequencing_prompts.append({
            "clean_prompt": f"{example_in}:{example_out}::{target_in}:",
            "clean_correct_answer": target_out,
            "clean_incorrect_answer": target_in,
            "corrupted_prompt": f"{example_in}:{example_in}::{target_in}:",
            "corrupted_correct_answer": target_in,
            "corrupted_incorrect_answer": target_out,
        })
    
    # # ===== CHANGE 8: POSITION CONSISTENCY =====
    # max_length = max(
    #     max(len(p['clean_prompt']) for p in analogy_prompts),
    #     max(len(p['clean_prompt']) for p in sequencing_prompts)
    # ) + 5
    
    # for dataset in [analogy_prompts, sequencing_prompts]:
    #     for p in dataset:
    #         p['clean_prompt'] = p['clean_prompt'].ljust(max_length)
    #         p['corrupted_prompt'] = p['corrupted_prompt'].ljust(max_length)
    # # ===== END CHANGE 8 =====

    datasets = {
        "analogy": {
            "description": "Letter-String Analogy Task ('+1' vs No Rule)",
            "prompts": analogy_prompts
        },
        "sequencing": {
            "description": "Next-Item Sequencing Task",
            "prompts": sequencing_prompts
        }
    }
    print(datasets)
    return datasets

# ===== CHANGE 4: LOCAL CACHE ISOLATION =====
def caching_hook_factory(cache: dict, hook_name: str) -> Callable:
    def hook(module, input, output):
        tensor_to_cache = output[0] if isinstance(output, tuple) else output
        cache[hook_name] = tensor_to_cache.detach().clone()
    return hook

def patching_hook_factory(cache: dict, hook_name: str, head_index: int = None, d_head: int = None) -> Callable:
    def hook(module, input, output):
        if hook_name not in cache:
            raise ValueError(f"Activation for {hook_name} not found!")
        cached_activation = cache[hook_name]
        patched_output = output.clone()
        min_seq_len = min(patched_output.shape[-2], cached_activation.shape[-2])
        if head_index is not None:
            start, end = head_index * d_head, (head_index + 1) * d_head
            if patched_output.ndim == 3: 
                patched_output[:, :min_seq_len, start:end] = cached_activation[:, :min_seq_len, start:end]
            elif patched_output.ndim == 2: 
                patched_output[:min_seq_len, start:end] = cached_activation[:min_seq_len, start:end]
        else:
            if patched_output.ndim == 3: 
                patched_output[:, :min_seq_len, :] = cached_activation[:, :min_seq_len, :]
            elif patched_output.ndim == 2: 
                patched_output[:min_seq_len, :] = cached_activation[:min_seq_len, :]
        return patched_output
    return hook
# ===== END CHANGE 4 =====

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for part in name.split('.'): model = getattr(model, part)
    return model

def run_with_hooks(model: nn.Module, tokenizer: AutoTokenizer, prompt: str, hook_fns: Dict[str, Callable]) -> torch.Tensor:
    handles = []
    try:
        for name, hook_fn in hook_fns.items():
            module = get_module_by_name(model, name)
            handles.append(module.register_forward_hook(hook_fn))
        
        # ===== CHANGE 7: ATTENTION MASK ENFORCEMENT =====
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move ALL to device
        # ===== END CHANGE 7 =====
        
        with torch.no_grad(): 
            outputs = model(**inputs)
        return outputs.logits[0, -1, :]
    finally:
        for handle in handles: handle.remove()

def calculate_logit_diff(logits: torch.Tensor, tokenizer: AutoTokenizer, correct_answer: str, incorrect_answer: str) -> float:
    try:
        # ===== CHANGE 3: TOKENIZATION SAFETY =====
        correct_tokens = tokenizer.encode(correct_answer, add_special_tokens=False)
        incorrect_tokens = tokenizer.encode(incorrect_answer, add_special_tokens=False)
        
        if not correct_tokens or not incorrect_tokens:
            return 0.0
            
        correct_id = correct_tokens[0]
        incorrect_id = incorrect_tokens[0]
        # ===== END CHANGE 3 =====

        return (logits[correct_id] - logits[incorrect_id]).item()
    except Exception as e:
        print(f"Error in calculate_logit_diff: {e}")
        return 0.0

def perform_patching_experiment(model: nn.Module, tokenizer: AutoTokenizer, model_config: Dict, 
                               source_prompt: str, dest_prompt: str, dest_correct_answer: str, 
                               dest_incorrect_answer: str, layer: int, component_type: str, 
                               head_index: int = None) -> float:
    # ===== CHANGE 4: LOCAL CACHE ISOLATION =====
    local_cache = {}
    # ===== END CHANGE 4 =====
    
    hook_template = model_config['mlp_hook_name_template'] if component_type == 'mlp' else model_config['attn_hook_name_template']
    hook_name = hook_template.format(layer)
    
    # ===== CHANGE 4: USE LOCAL CACHE =====
    run_with_hooks(model, tokenizer, source_prompt, 
                  {hook_name: caching_hook_factory(local_cache, hook_name)})
    # ===== END CHANGE 4 =====
    
    d_head = model_config["d_model"] // model_config["n_heads"] if component_type == 'attn_head' else None
    
    # ===== CHANGE 4: USE LOCAL CACHE =====
    patching_hooks = {hook_name: patching_hook_factory(local_cache, hook_name, head_index, d_head)}
    # ===== END CHANGE 4 =====
    
    patched_logits = run_with_hooks(model, tokenizer, dest_prompt, patching_hooks)
    return calculate_logit_diff(patched_logits, tokenizer, dest_correct_answer, dest_incorrect_answer)

def run_exploratory_sweep(model: nn.Module, tokenizer: AutoTokenizer, model_config: Dict, task_data: Dict, patch_type: str) -> pd.DataFrame:
    n_layers, n_heads = model_config["n_layers"], model_config["n_heads"]
    prompt_dataset = task_data['prompts']

    print(f"\nRunning {patch_type} sweep for '{task_data['description']}' over {len(prompt_dataset)} prompts...")

    clean_baselines, corrupted_baselines = [], []
    for prompt_set in prompt_dataset:
        clean_logits = run_with_hooks(model, tokenizer, prompt_set['clean_prompt'], {})
        clean_baselines.append(calculate_logit_diff(clean_logits, tokenizer, prompt_set['clean_correct_answer'], prompt_set['clean_incorrect_answer']))
        corrupted_logits = run_with_hooks(model, tokenizer, prompt_set['corrupted_prompt'], {})
        corrupted_baselines.append(calculate_logit_diff(corrupted_logits, tokenizer, prompt_set['corrupted_correct_answer'], prompt_set['corrupted_incorrect_answer']))

    results = []
    for component_type, head_range in [('attn_head', range(n_heads)), ('mlp', [-1])]:
        print(f"  - Patching {component_type}s...")
        for layer in range(n_layers):
            for head_index in head_range:
                effects_for_this_component = []
                for i, prompt_set in enumerate(prompt_dataset):
                    if patch_type == 'noising':
                        source_prompt, dest_prompt = prompt_set['corrupted_prompt'], prompt_set['clean_prompt']
                        dest_correct, dest_incorrect = prompt_set['clean_correct_answer'], prompt_set['clean_incorrect_answer']
                        baseline_to_compare = clean_baselines[i]
                    else: # denoising
                        source_prompt, dest_prompt = prompt_set['clean_prompt'], prompt_set['corrupted_prompt']
                        dest_correct, dest_incorrect = prompt_set['corrupted_correct_answer'], prompt_set['corrupted_incorrect_answer']
                        baseline_to_compare = corrupted_baselines[i]

                    patched_logit_diff = perform_patching_experiment(
                        model, tokenizer, model_config, source_prompt, dest_prompt, 
                        dest_correct, dest_incorrect, layer, component_type, 
                        head_index if component_type == 'attn_head' else None
                    )
                    effect = patched_logit_diff - baseline_to_compare
                    
                    effects_for_this_component.append(effect)

                # ===== CHANGE 5: ROBUST AGGREGATION =====
                average_effect = np.median(effects_for_this_component)  # Use median instead of mean
                # ===== END CHANGE 5 =====
                
                results.append({'layer': layer, 'head': head_index, 'type': component_type, 'effect': average_effect})

    return pd.DataFrame(results)

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

def main():
    models_to_test = {
    'gpt2': 'gpt2',  # Uncommented
    #'gptneo': 'EleutherAI/gpt-neo-125m',
    'gptj6b': 'EleutherAI/gpt-j-6B',
    'llama27b': 'meta-llama/Llama-2-7b-hf',
    'llama213b': 'meta-llama/Llama-2-13b-hf',
    'gptneox20b': 'EleutherAI/gpt-neox-20b',
    #'llama270b': 'meta-llama/Llama-2-70b-hf'
    }

    main_output_dir = "Results_8_New_WithChanges"
    os.makedirs(main_output_dir, exist_ok=True)
    datasets = get_task_datasets()

    for model_short_name, model_hf_name in models_to_test.items():
        torch.cuda.empty_cache()
        try:
            model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_hf_name)
        except Exception as e:
            print(f"\n--- Could not load model {model_hf_name}. Skipping. Error: {e} ---\n")

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

            continue

        model_results_dir = os.path.join(main_output_dir, model_short_name)
        os.makedirs(model_results_dir, exist_ok=True)

        for task_name, task_data in datasets.items():
            for patch_type in ['noising']:
                result_key = f"{task_name}_{patch_type}"
                df = run_exploratory_sweep(model, tokenizer, model_config, task_data, patch_type)

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