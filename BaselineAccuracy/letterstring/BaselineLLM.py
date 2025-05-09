import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *


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

            access_token = "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"

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
            access_token = "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"

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


def generate_response(prompt, model, tokenizer, max_new_tokens=16):
    """
    Generate a response from the model for the given prompt

    Parameters:
    prompt: Input text prompt
    model: HuggingFace model
    tokenizer: HuggingFace tokenizer
    max_new_tokens: Maximum new tokens to generate

    Returns:
    response: Generated text response
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the output, skipping the input
    response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response


model_name = 'gpt2'
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

prompt = "Once upon a time"
#inputs = tokenizer(prompt, return_tensors="pt")

#co = outputLLM(prompt, model, model_config, tokenizer, max_new_tokens=16)

#print("Output:", repr(tokenizer.decode(co.squeeze())))

print(generate_response(prompt,model,tokenizer))

"""
outputs = model.generate(**inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
"""

gen = 'nogen'
num_permuted = 1
promptstyle = 'hw'

all_prob = np.load(f'./problems/{gen}/all_prob_{num_permuted}_7_human.npz', allow_pickle=True)['all_prob']

response_dict={}

for alph in all_prob.item().keys():
    prob_types = ['succ']
    N_prob_types = len(prob_types)

    N_trials_per_prob_type = 10
    all_prob_type_responses = []
    count = 0

    for p in range(N_prob_types):
        prob_type_responses = []

        for t in range(N_trials_per_prob_type):
            prob = all_prob.item()[alph][prob_types[p]]['prob'][t]
            prompt=''

            if promptstyle == 'hw':
                prompt += "Let's try to complete the pattern:\n\n"

                prompt += '['
                for i in range(len(prob[0][0])):
                    prompt += str(prob[0][0][i])
                    if i < len(prob[0][0]) - 1:
                        prompt += ' '
                prompt += '] ['
                for i in range(len(prob[0][1])):
                    prompt += str(prob[0][1][i])
                    if i < len(prob[0][1]) - 1:
                        prompt += ' '
                prompt += ']\n['
                for i in range(len(prob[1][0])):
                    prompt += str(prob[1][0][i])
                    if i < len(prob[1][0]) - 1:
                        prompt += ' '
                if promptstyle in ["minimal", "hw", "webb","webbplus"]:
                    prompt += '] ['
                else:
                    prompt += '] [ ? ]'

            response = []

            while len(response) == 0:
                response = generate_response(prompt,model,tokenizer)
                prob_type_responses.append(response)
            count += 1

        all_prob_type_responses.append(prob_type_responses)

        response_dict[alph] = all_prob_type_responses

        path = f'{model_name}_prob_predictions_multi_alph/{gen}'

        check_path(path)

        save_fname = f'./{path}/{model_name}_letterstring_results_{num_permuted}_multi_alph_gptprobs'

        save_fname += '.npz'
        np.savez(save_fname, all_prob_type_responses=response_dict, allow_pickle=True)


zero_gen_probs = ['succ']

num_alphs = 7
perms = [1]
models = ['gpt2']
promptstyles = ['hw']
nperms = [f'np_{i}' for i in perms]
alphs = [f'alph_{j}' for j in range(num_alphs)]
prob_types = ['succ']

for gpt in models:
    gpt_dict[gpt] = dict()
    for promptstyle in promptstyles:
        gpt_dict[gpt][promptstyle] = dict()
        for i in perms:
            if promptstyle in ['webb', 'webbplus'] and i != 1:
                continue
            prob_npz = np.load(f'../problems/human/all_prob_{i}_{num_alphs}_human.npz', allow_pickle=True)
            prob_dict[f'np_{i}'] = prob_npz['all_prob'].item()
            gpt_npz = np.load(f'../GPT{gpt}_prob_predictions_multi_alph/nogen/gpt{gpt}_letterstring_results_{i}_multi_alph_{promptstyle}.npz', allow_pickle=True)
            gpt_dict[gpt][promptstyle][f'np_{i}'] = gpt_npz['all_prob_type_responses'].item()
            for k, alph in enumerate(prob_dict[f'np_{i}'].keys()):
                if k > 6:
                    continue
                current_prob_types = list(prob_dict[f'np_{i}'][alph].keys())
                current_prob_types.remove('shuffled_letters')
                current_prob_types.remove('shuffled_alphabet')
                for j, prob_type in enumerate(current_prob_types):
                    for prob_ind in range(len(prob_dict[f'np_{i}'][alph][prob_type]['prob'])):
                        prob = prob_dict[f'np_{i}'][alph][prob_type]['prob'][int(prob_ind)]
                        source = prob[0]
                        target = prob[1]
                        if prob_type == 'attn':
                            # print(rs)
                            correct_answer = ['a', 'a', 'a', 'a']
                        else:
                            correct_answer = prob[1][1]

                        gpt_response = gpt_dict[gpt][promptstyle][f'np_{i}'][alph][j][prob_ind]
                        first_bracket = gpt_response.find(']')
                        if first_bracket != -1:
                            gpt_response = gpt_response[:first_bracket]
                        given_answer = [a for a in list(gpt_response) if a not in ' []']
                        if len(correct_answer) != len(given_answer):
                            incorrect = 1
                        else:
                            incorrect = sum([a!=b for a, b in zip(correct_answer, given_answer)])
                        correct = not incorrect

                        total = 1
                        if promptstyle =='human':
                            data.append([f'gpt_{gpt}', gpt, 'human-like', f'np_{i}', alph, prob_type, prob_ind, source[0], source[1], target[0], correct_answer, given_answer, correct, total])
                        else:
                            data.append([f'gpt_{gpt}', gpt, promptstyle, f'np_{i}', alph, prob_type, prob_ind, source[0], source[1], target[0], correct_answer, given_answer, correct, total])

