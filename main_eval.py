from convokit import Corpus, download, Conversation
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance, corpus_len, corpus_to_prompts
from toxicity.split_data import filter_corpus_by_file


from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed
from evaluator import Evaluator

import random
import argparse
import json

import logging

from accelerate.logging import get_logger
from accelerate.utils.tqdm import tqdm

from datasets import Dataset

import os

os.environ["WANDB_PROJECT"] = "ast"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

logger = get_logger("ast")

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.ERROR)
logger.setLevel(logging.DEBUG)


# fix random sede for reproducibility
R = random.Random(24)
# to test if its data dependent
# R = random.Random(54)

# TEACH = False

# if not TEACH:
# load our initial corpus ahead of time
# test_corp = filter_corpus_by_file(Corpus(filename=download("reddit-corpus-small")),
#                                   "data/test.txt")
# prompts = corpus_to_prompts(test_corp)
# print(len(prompts))

# fire this puppy off 
if __name__ == "__main__":
    # set random seed for reproducability
    set_seed(24)

    # establish the arguments of this system
    parser = argparse.ArgumentParser(description='AST Evaluator')
    parser.add_argument('weights', type=str, help='which model shall we evaluate?')
    parser.add_argument('--defense', type=str, help='what weights should we use for defense?',
                        default= "meta-llama/Llama-3.1-8B") #"openai-community/gpt2"
    parser.add_argument('--horizon', type=int, help='how many turns to self-play?',
                        default=5)
    parser.add_argument('--save_dir', type=str, default='results',
                        help='prefix of the model save dir, default "results"')
    parser.add_argument('--save_name', type=str, default=None,
                        help='what to name the results')
    parser.add_argument('--prompt_type', type=str, default='reddit',
                        help='what category of prompts to initiate evals? options: reddit, wiki, code, news')

    args = parser.parse_args()
    #len(prompts = 466) -> keep consistent
    
    evaluator = Evaluator(args)
    
    if args.prompt_type == "reddit":
        prompts = test_corp = filter_corpus_by_file(Corpus(filename=download("reddit-corpus-small")),"data/test.txt")
        prompts = corpus_to_prompts(test_corp)
        #print(len(prompts))
        dl = evaluator.load(prompts)


    else:
        dl = evaluator.load_custom(args.prompt_type, num_samples=466)
    evaluator(dl)


