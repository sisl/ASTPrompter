from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
from datasets import load_dataset
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from environment import episode
import torch
import random
import os
import json

from rich.console import Console
from rich.text import Text
from tqdm import tqdm
from pynvml import *

# Used to get GPU information globally (for processes other than the current one)
# Used to inform GPU selection
nvmlInit()

def get_free_gpu():
    if not torch.cuda.is_available():
        return "cpu"
    free_memory = []
    
    for i in range(torch.cuda.device_count()):
        gpu_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
        free_memory.append((gpu_info.free, i))
    _, best_gpu = max(free_memory)
    return f"cuda:{best_gpu}"


attacker_name = "llama_v_llama_best"
checkpoint = os.getenv("HOME") + "/models/" + attacker_name
base_name = "TinyLlama/TinyLlama_v1.1"
defender_name = "TinyLlama/TinyLlama_v1.1"
device = get_free_gpu()

# prompt_source = "reddit" # Options: real-toxicity-prompts, reddit
prompt_source = "real-toxicity-prompts"

max_prompts = 6000

#  Select and 
if prompt_source == "reddit":
    corpus = Corpus(filename=download("reddit-corpus-small"))
    id2results = jsonl_to_dict('detox_results.jsonl')
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)
    convos = list(corpus.conversations.values())

    # we only keep the last five utterances (and also discard the front
    # because the front is the self-post on reddit)
    prompts = [[clean_utterance(j.text)
                for j in list(i.iter_utterances())
                if j.text.strip() != "[deleted]"
                and j.text.strip() != ""][1:][-2:]
            for i in convos]
    prompts = [[j+" " for j in i if j.strip() != ""]
            for i in prompts]
    prompts = [i for i in prompts if len(i) != 0]
elif prompt_source == "real-toxicity-prompts":
    ds = load_dataset("allenai/real-toxicity-prompts")

    prompts = [p['text'] for p in ds['train']['prompt']]
else:
    raise ValueError("Invalid prompt source")

model = AutoModelForCausalLM.from_pretrained(checkpoint)
model_base = AutoModelForCausalLM.from_pretrained(base_name)
# model_defender = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="flash_attention_2", load_in_4bit=True, torch_dtype=torch.float16)
model_defender = AutoModelForCausalLM.from_pretrained(defender_name)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
tokenizer_defender = AutoTokenizer.from_pretrained(defender_name)

adversary = LanguageModel(dont_init=True)
adversary.model = model
adversary.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
adversary.to(torch.device(device))

base = LanguageModel(dont_init=True)
base.model = model_base
base.tokenizer = tokenizer
base.to(torch.device(device))

defender = LanguageModel(dont_init=True)
defender.model = model_defender
defender.tokenizer = tokenizer_defender
defender.to(torch.device(device))

rollouts_per_prompt = 5
turns = 1

output_data = {
    "config": {
        "adversary": attacker_name,
        "base": base_name,
        "defender": defender_name,
        "rollouts_per_prompt": rollouts_per_prompt,
        "turns": turns,
        "prompt_source": prompt_source
    },
    "rollouts": []
}

for prompt_id, prompt in enumerate(tqdm(prompts[:min(max_prompts, len(prompts))], desc="Rolling out prompts")):
    if prompt_source == "reddit":
        # Ensure prompt is just a single utterance if from Reddit
        p = [" ".join(prompt).replace('\n', ' ')] 
    else:
        p = [prompt]

    for _ in range(rollouts_per_prompt):
        ro_policy = episode(adversary, defender, p, horizon=turns, return_sequence=True)
        ro_base = episode(base, defender, p, horizon=turns, return_sequence=True)

        output_data["rollouts"].append({
            "prompt_id": prompt_id,
            "prompt": p,
            "attacker_rollout": ro_policy,
            "baseline_rollout": ro_base
        })

# Check that output/ exists and create it if it doesn't
if not os.path.exists("output"):
    os.makedirs("output")

json.dump(output_data, open(f"output/rollout_data_{attacker_name}_{prompt_source.replace("-", "_")}.json", "w"))