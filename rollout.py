from lm import *
from environment import *
from convokit import Corpus, download, Conversation
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance
from datasets import Dataset

from tqdm import tqdm

BASE_MODEL = "openai-community/gpt2"
DEFENDER_MODEL = "openai-community/gpt2"
SAVE_PATH = "./data/base.hf"


# load our initial corpus ahead of time
corpus = Corpus(filename=download("reddit-corpus-small"))
id2results = jsonl_to_dict('detox_results.jsonl')
corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
corpus = filter_corpus_formatting(corpus)
convos = list(corpus.conversations.values())

adversary = LanguageModel(BASE_MODEL).to("cuda")
defender = LanguageModel(DEFENDER_MODEL).to("cuda")

defender.model.eval()


def gen():
    for i in tqdm(convos):
        samples = episode_paired(adversary, defender, i, device="cuda")
        for j in samples:
            yield {
                "prompt": j.query,
                "chosen": j.response_w,
                "rejected": j.response_l,
            }

dataset = Dataset.from_generator(gen)
dataset.save_to_disk(SAVE_PATH)

