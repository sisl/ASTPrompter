from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from environment import episode
import torch
import random

checkpoint = "./models/dpo_model_gpt2_teachmix_checkpoint"

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
model_base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
# model_defender = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="flash_attention_2", load_in_4bit=True, torch_dtype=torch.float16)
model_defender = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer_defender = AutoTokenizer.from_pretrained("openai-community/gpt2")

adversary = LanguageModel(dont_init=True)
adversary.model = model
adversary.tokenizer = tokenizer

base = LanguageModel(dont_init=True)
base.model = model_base
base.tokenizer = tokenizer

defender = LanguageModel(dont_init=True)
defender.model = model_defender
defender.tokenizer = tokenizer_defender


while True:
    prompt = []
    r = None
    while r != 'q':
        r = input("> ").strip()
        if r == "redditme":
            prompt = random.choice(prompts)
            print("==== PROMPT ====")
            print(" ".join(prompt))
            break
        if r != "q":
            prompt.append(r)

    ro_policy = episode(adversary, defender, prompt, horizon=5, return_sequence=True)
    ro_base = episode(base, defender, prompt, horizon=5, return_sequence=True)

    print("==== POLICY ====")
    print("".join("["+i+"] " for i in ro_policy))
    print("==== BASE ====")
    print("".join("["+i+"] " for i in ro_base))

    breakpoint()


