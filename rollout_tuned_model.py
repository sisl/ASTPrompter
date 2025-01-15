from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
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

attacker_name = "/models/llama_v_llama_best"
checkpoint = os.getenv("HOME") + attacker_name
base_name = "TinyLlama/TinyLlama_v1.1"
defender_name = "TinyLlama/TinyLlama_v1.1"

# load our initial corpus ahead of time
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

model = AutoModelForCausalLM.from_pretrained(checkpoint)
model_base = AutoModelForCausalLM.from_pretrained(base_name)
# model_defender = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="flash_attention_2", load_in_4bit=True, torch_dtype=torch.float16)
model_defender = AutoModelForCausalLM.from_pretrained(defender_name)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
tokenizer_defender = AutoTokenizer.from_pretrained(defender_name)

adversary = LanguageModel(dont_init=True)
adversary.model = model
adversary.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
adversary.to(torch.device("cuda"))

base = LanguageModel(dont_init=True)
base.model = model_base
base.tokenizer = tokenizer
base.to(torch.device("cuda"))

defender = LanguageModel(dont_init=True)
defender.model = model_defender
defender.tokenizer = tokenizer_defender
defender.to(torch.device("cuda"))

rollouts_per_prompt = 5
turns = 2

output_data = {
    "config": {
        "adversary": attacker_name,
        "base": base_name,
        "defender": defender_name,
        "rollouts_per_prompt": rollouts_per_prompt,
        "turns": turns
    },
    "rollouts": []
}

for prompt_id, prompt in enumerate(tqdm(prompts, desc="Rolling out prompts")):
    p = [" ".join(prompt).replace('\n', ' ')]  # Ensure prompt is just a single utterance

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

json.dump(output_data, open("output/rollout_data_llama_v_llama.json", "w"))