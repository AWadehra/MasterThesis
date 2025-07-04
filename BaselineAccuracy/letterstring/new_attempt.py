import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import json
import random
from typing import *
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def load_gpt_model_and_tokenizer(model_name:str, device='cuda'):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'

    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names

    """
    assert model_name is not None

    print("Loading: ", model_name)

    if 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}

    elif 'gpt2' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}

    elif 'gpt-neo-125m' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        MODEL_CONFIG={"n_heads":model.config.num_heads,
                      "n_layers":model.config.num_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neo.layers.{layer}.attention.dense' for layer in range(model.config.num_layers)],
                      "layer_hook_names":[f'gpt_neo.layers.{layer}' for layer in range(model.config.num_layers)],
                      "prepend_bos":False}

    elif 'gpt-neox' in model_name.lower() or 'pythia' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                      "prepend_bos":False}

    elif 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            access_token = "hf_findNewOne"

            tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token)
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                token=access_token
            )
        else:
            if '7b' in model_name.lower() or '8b' in model_name.lower():
                model_dtype = torch.float32
            else: #half precision for bigger llama models
                #This becomes only for the 13B model then. Okay then. What else?
                model_dtype = torch.float16

            # If transformers version is < 4.31 use LlamaLoaders
            # tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

            # If transformers version is >= 4.31, use AutoLoaders
            access_token = "hf_findNewOne"

            tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, token=access_token).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                      "prepend_bos":True}
    else:
        raise NotImplementedError("Still working to get this model available!")


    return model, tokenizer, MODEL_CONFIG

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def generate_response(prompt, model, tokenizer, max_new_tokens=40):
    """
    Generate a response from the model for the given prompt
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def outputLLM(sentence, model, model_config, tokenizer, max_new_tokens=16):
    """
    Allows for intervention in natural text where we generate and intervene on several tokens in a row.

    Parameters:
    sentence: sentence to intervene on with the FV
    edit_layer: layer at which to add the function vector
    function_vector: vector to add to the model that triggers execution of a task
    model: huggingface model
    model_config: dict with model config parameters (n_layers, n_heads, etc.)
    tokenizer: huggingface tokenizer
    max_new_tokens: number of tokens to generate
    num_interv_tokens: number of tokens to apply the intervention for (defaults to all subsequent generations)
    do_sample: whether to sample from top p tokens (True) or have deterministic greedy decoding (False)

    Returns:
    clean_output: tokens of clean output
    intervention_output: tokens of intervention output

    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    clean_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    return clean_output

def load_problems(num_permuted=1):
    """
    Load nogen problems without shuffled letters
    """
    prob_path = f'./problems/nogen/all_prob_{num_permuted}_7_human.npz'
    if not os.path.exists(prob_path):
        raise FileNotFoundError(f"Problem file not found at {prob_path}")

    all_prob = np.load(prob_path, allow_pickle=True)['all_prob'].item()

    # Filter out problems with shuffled letters
    filtered_probs = {}
    for alph in all_prob.keys():
        if all_prob[alph]['shuffled_letters'] is None:
            filtered_probs[alph] = all_prob[alph]

    return filtered_probs

def evaluate_model(model, tokenizer, problems, promptstyle='hw'):
    """
    Evaluate model on all problems and return accuracy results
    """
    data = []
    prob_types = ['succ']

    for alph in problems.keys():
        shuffled_alphabet = list(problems[alph]['shuffled_alphabet'])
        alph_string = ' '.join(shuffled_alphabet)

        for prob_type in prob_types:
            if prob_type not in problems[alph]:
                continue

            for prob_ind in range(len(problems[alph][prob_type]['prob'])):
                prob = problems[alph][prob_type]['prob'][prob_ind]

                # Generate prompt
                prompt = ''
                if promptstyle == 'hw':
                    #prompt += 'Use this fictional alphabet: \n\n' \
                    #          + alph_string \
                    #          + "\n\nLet's try to complete the pattern:\n\n"

                    prompt += "Let's try to complete the pattern:\n\n"

                # Format problem
                prompt += '[' + ' '.join(prob[0][0]) + '] [' + ' '.join(prob[0][1]) + ']\n[' \
                          + ' '.join(prob[1][0]) + '] [' + ' '.join(prob[1][1][:-1])

                # Get model response
                response = generate_response(prompt, model, tokenizer)

                # Process response
                first_bracket = response.find(']')
                if first_bracket != -1:
                    response = response[:first_bracket]
                given_answer = [a for a in list(response) if a not in ' []']

                # Get correct answer
                if prob_type == 'attn':
                    correct_answer = ['a', 'a', 'a', 'a']
                else:
                    correct_answer = prob[1][1][-1]

                # Check correctness
                if len(correct_answer) != len(given_answer):
                    incorrect = 1
                else:
                    incorrect = sum([a!=b for a, b in zip(correct_answer, given_answer)])
                correct = not incorrect

                data.append({
                    'alph': alph,
                    'prob_type': prob_type,
                    'prob_ind': prob_ind,
                    'source_1': prob[0][0],
                    'source_2': prob[0][1],
                    'target_1': prob[1][0],
                    'correct_answer': correct_answer,
                    'given_answer': given_answer,
                    'correct': correct
                })

    return pd.DataFrame(data)

def calculate_accuracy(results_df, model_name):
    """
    Calculate and print accuracy statistics
    """
    # Overall accuracy
    overall_acc = results_df['correct'].mean()
    ci_low, ci_high = proportion_confint(sum(results_df['correct']), len(results_df))
    print(f"Overall accuracy: {overall_acc:.3f} ({ci_low:.3f}-{ci_high:.3f})")


    """
    # Accuracy by problem type
    print("\nAccuracy by problem type:")
    for prob_type, group in results_df.groupby('prob_type'):
        acc = group['correct'].mean()
        ci_low, ci_high = proportion_confint(sum(group['correct']), len(group))
        print(f"{prob_type}: {acc:.3f} ({ci_low:.3f}-{ci_high:.3f})")
    """

    data = {'Accuracy': overall_acc, 'CI_low': ci_low, 'CI_high': ci_high, 'model_name' : model_name }

    df = pd.DataFrame([data])

    return df

def save_results(accuracy_stats, model_name, subfolder_name, output_dir='results'):
    """
    Save evaluation results to files
    """
    full_path = os.path.join(output_dir, subfolder_name)
    os.makedirs(full_path, exist_ok=True)

    # Save raw results
    filename = f'newtype_{model_name}.csv'
    accuracy_stats.to_csv(os.path.join(full_path, filename), index=False)

    #print(f"Results saved to {output_dir}")

def main():
    # Load model

    models = {
                'gpt2':'gpt2',
              'gptneo': 'EleutherAI/gpt-neo-125m'
               }

    for model_name, model_technical_name in models.items():
        #model_technical_name = 'gpt2'
        torch.cuda.empty_cache()  # Clear cache before loading new model
        model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_technical_name)

        # Load problems (unpermuted alphabet)
        problems = load_problems(num_permuted=1)

        # Evaluate model
        results_df = evaluate_model(model, tokenizer, problems)

        # Calculate accuracy
        accuracy_stats = calculate_accuracy(results_df, model_name)

        # Save results
        save_results(accuracy_stats, model_name)

        # Clean up
        del model, tokenizer, model_config
        torch.cuda.empty_cache()  # Clear cache after evaluation

if __name__ == "__main__":
    main()