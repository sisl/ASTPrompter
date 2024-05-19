from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from environment import episode
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
prompts = [[j for j in i if j.strip() != ""]
        for i in prompts]
prompts = [i for i in prompts if len(i) != 0]

model = AutoModelForCausalLM.from_pretrained(checkpoint)
model_base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

adversary = LanguageModel(dont_init=True)
adversary.model = model
adversary.tokenizer = tokenizer

defender = LanguageModel(dont_init=True)
defender.model = model_base
defender.tokenizer = tokenizer

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

    ro_policy = episode(adversary, defender, [i+" " for i in prompt], horizon=3)
    ro_base = episode(defender, defender, [i+" " for i in prompt], horizon=3)

    print("==== POLICY ====")
    print(ro_policy[-1].query+ro_policy[-1].response)
    print("==== BASE ====")
    print(ro_base[-1].query+ro_base[-1].response)

    breakpoint()


