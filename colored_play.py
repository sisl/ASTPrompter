from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from convokit import Corpus, download, Conversation
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from environment import episode
import torch
import random
import os

from rich.console import Console
from rich.text import Text

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

checkpoint = os.getenv("HOME") + "/models/llama_v_llama_best"
base = "TinyLlama/TinyLlama_v1.1"
defender = "TinyLlama/TinyLlama_v1.1"
device = get_free_gpu()

# prompt_source = "reddit" # Options: real-toxicity-prompts, reddit
prompt_source = "real-toxicity-prompts"

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
model_base = AutoModelForCausalLM.from_pretrained(base)
# model_defender = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="flash_attention_2", load_in_4bit=True, torch_dtype=torch.float16)
model_defender = AutoModelForCausalLM.from_pretrained(defender)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
tokenizer_defender = AutoTokenizer.from_pretrained(defender)

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


console = Console()

prompt = random.choice(prompts)

if prompt_source == "reddit":
    prompt = [" ".join(prompt).replace('\n', ' ')] # Ensure prompt is just a single utterance
else:
    prompt = [prompt]

prompt_text = prompt[0] # Save the original prompt text for further display
console.print("==== PROMPT ====", style="bold")
console.print(Text(prompt_text))

ro_policy = episode(adversary, defender, prompt, horizon=5, return_sequence=True)
ro_base = episode(base, defender, prompt, horizon=5, return_sequence=True)

# Pop off the first entry in the rollout (the propmpt)
ro_policy = ro_policy[1:]
ro_base = ro_base[1:]

ro_policy_attacker = [i.replace('\n', ' ') for i in ro_policy[0::2]]
ro_policy_defender = [i.replace('\n', ' ') for i in ro_policy[1::2]]

ro_base_attacker = [i.replace('\n', ' ') for i in ro_base[0::2]]
ro_base_defender = [i.replace('\n', ' ') for i in ro_base[1::2]]

policy_text = Text(prompt_text)
base_text = Text(prompt_text)

for i in range(len(ro_policy_attacker)):
    policy_text.append(ro_policy_attacker[i], style="red")
    policy_text.append(ro_policy_defender[i], style="blue")

for i in range(len(ro_base_attacker)):
    base_text.append(ro_base_attacker[i], style="red")
    base_text.append(ro_base_defender[i], style="blue")

console.print("==== TUNED ATTACKER ====", style="bold")
console.print(policy_text)
console.print("==== BASELINE ====", style="bold")
console.print(base_text)

