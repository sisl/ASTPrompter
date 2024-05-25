"""
Run detoxify model on Reddit Conversation Corpus (RCC) and output
filtered dataset compatible with RCC Corpus class based on tunable 
toxicity threshold. 
"""
from convokit import Corpus, download, Conversation
from detoxify import Detoxify
import json
import jsonlines
import random
from tqdm import tqdm


def detoxify_selector(id2results:dict[str], attribute2thresh:dict[str], convo:Conversation):
    """Selector function to filter corpus

    Parameters
    ----------
    id2results : dict[str]
        dict mapping conversation ids to results from the detoxify model. 
        detoxify model should be run in advance to populate this dict to 
        avoid delays from time needed for inference. (see score_corpus
        and json_to_dict)
    attribute2thresh : dict[str]
        dict mapping attribute names to acceptable scores from detoxify model
        For the 'original' model, possible attributes are:
        - toxicity
        - severe_toxicity
        - obscene
        - threat
        - insult
        - identity_attack
    convo : Conversation
        The conversation to assess

    Returns
    -------
    bool
        Whether or not the conversation should remain in the corpus
    """
    id = str(convo.id)
    if id not in id2results.keys():
        return False
    results = id2results[id]

    for attribute in attribute2thresh.keys():
        if results[attribute] > attribute2thresh[attribute]:
            return False
    return True

def filter_corpus_toxicity(corpus:Corpus, id2results:dict[str], attribute2thresh:dict[str]={"toxicity": 0.5}):
    return corpus.filter_conversations_by(lambda x: detoxify_selector(id2results, attribute2thresh, x))

def conv_to_str(convo:Conversation):
    convo = [i.text for i in convo.iter_utterances()][1:]
    convo = [f"anon{int(i % 2 == 0)}: {text.strip()}" for i, text in enumerate(convo)]
    convo_str =  "\n".join(convo).strip()
    return convo_str

# Detoxify options are 'original', 'unbiased', and 'multilingual'
# See: https://github.com/unitaryai/detoxify/tree/master?tab=readme-ov-file#prediction
def score_corpus(corpus:Corpus, out_fname:str, detoxify_model:str="original"):
    with jsonlines.open(out_fname, 'w', flush=True) as writer:
        for convo in tqdm(corpus.iter_conversations()):
            convo_str = conv_to_str(convo)
            detox_results = Detoxify(detoxify_model).predict(convo_str)
            detox_results = {key: float(value) for key, value in detox_results.items()}
            conv_dict = {
                'id': str(convo["id"]),
                'convo_str': convo_str,
                'detox_results': json.dumps(detox_results)
            }
            writer.write(conv_dict)

def jsonl_to_dict(fname):
    id2results = {}
    with jsonlines.open(fname) as reader:
        for obj in reader.iter(type=dict):
            id = obj['id']
            results = obj['detox_results']
            id2results[id] = json.loads(results)
    return id2results


if __name__ == "__main__":
    # download large conversation corpus
    corpus = Corpus(filename=download("reddit-corpus-small"))
    #score_corpus(corpus, 'detox_results.jsonl')
    id2results = jsonl_to_dict('detox_results.jsonl')
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
