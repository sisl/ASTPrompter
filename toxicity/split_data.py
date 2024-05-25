"""
Split convokit data into train/dev/test (80/05/15)
"""
from convokit import Corpus, download, Conversation
import json
import jsonlines
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .reddit_data_helpers import filter_corpus_formatting, corpus_len
from .detoxify_reddit import jsonl_to_dict, filter_corpus_toxicity

def filter_corpus_by_file(corpus:Corpus, fname:str):
    with open(fname, "r") as f:
        data = f.read()
        ids = data.strip().split()
        return corpus.filter_conversations_by(lambda x: x.id in ids)

def split_corpus(corpus:Corpus, train_fname:str, dev_fname:str, test_fname:str):
    # First filter corpus
    id2results = jsonl_to_dict('detox_results.jsonl')
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)

    # Now split corpus into train/dev/test
    # 80/05/15
    corpus_list = [conv for conv in corpus.iter_conversations()]
    train, dev_test = train_test_split(corpus_list, train_size=0.8, random_state=24)
    # 0.25 * 0.2 = 0.05
    dev, test = train_test_split(dev_test, train_size=0.25, random_state=24)
    
    # Write results to files for reproducibility
    with open(train_fname, "w") as train_file:
        for conv in train:
            train_file.write(str(conv.id)+" ")

    with open(dev_fname, "w") as dev_file:
        for conv in dev:
            dev_file.write(str(conv.id)+" ")
    
    with open(test_fname, "w") as test_file:
        for conv in test:
            test_file.write(str(conv.id)+" ")

    # Print to sanity check
    print("total examples post filtering",len(corpus_list))
    print("# examples train",len(train))
    print("# examples dev",len(dev))
    print("# examples test",len(test))

if __name__ == "__main__":
    # download large conversation corpus
    corpus = Corpus(filename=download("reddit-corpus-small"))

    split_corpus(corpus, "data/train.txt", "data/dev.txt", "data/test.txt")
    train_corpus = filter_corpus_by_file(corpus, "data/train.txt")
    print(corpus_len(train_corpus))
