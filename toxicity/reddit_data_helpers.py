"""
Helper functions for Reddit Conversation Corpus (RCC) to filter
toxicity and correct formatting.
"""
from convokit import Corpus, Conversation, download
from bs4 import BeautifulSoup
from html import unescape
from detoxify import Detoxify
import json
import jsonlines
import re
from tqdm import tqdm
import random

# fix random sede for reproducibility
R = random.Random(24)

"""
Address formatting issues in RCC
"""
def format_selector(convo:Conversation):
    """Selector function to filter corpus based on containing full transcripts
    We remove conversations containing strings in unallowed_strs since these
    indicate that part of the original conversation text was removed

    Parameters
    ----------
    convo : Conversation
        The conversation to evaluate

    Returns
    -------
    bool
        Whether or not the conversation should remain in the corpus
    """
    unallowed_strs = ["[removed]", "[deleted]", "[repost of same comment from that other thread]"]
    for utt in convo.iter_utterances():
        for unallowed in unallowed_strs:
            if unallowed in utt.text:
                return False
    return True

def filter_corpus_formatting(corpus:Corpus):
    """
    Remove conversations containing prohibited strings
    """
    return corpus.filter_conversations_by(format_selector)

def replace_matches(utterance:str, matches:list):
    """Replace match[0] with match[1]

    Helper fn to avoid repeating code
    """
    for match in matches:
        utterance = utterance.replace(match[0], match[1])
    return utterance

def format_urls(utterance:str):
    """Remove URLs from utterance but preserve text which pointed to them

    Parameters
    ----------
    utterance : str
        The string to remove URLs from
    
    Returns
    -------
        String without URLs
    """
    pattern = re.compile(r"(\[(.*?)\]\(.*?\))")
    matches = pattern.findall(utterance)
    return replace_matches(utterance, matches)

def remove_emojis(utterance:str):
    """Remove emojis from utterance
    
    From: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

    Parameters
    ----------
    utterance : str
        The string to remove emojis from
    
    Returns
    --------
        String without emojis
    """
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', utterance)

def strip_bold(utterance:str):
    """Remove bold formatting from utterance
    """
    bold = re.compile("(\*\*(.*?)\*\*)")
    matches = bold.findall(utterance)
    return replace_matches(utterance, matches)

def strip_italic(utterance:str):
    """Remove italic formatting from utterance
    """
    # Italic with double _
    italic_1 = re.compile("(__(.*?)__)")
    matches = italic_1.findall(utterance)
    utterance = replace_matches(utterance, matches)

    # Italic with single _
    italic_2 = re.compile("(_(.*?)_)")
    matches = italic_2.findall(utterance)
    return replace_matches(utterance, matches)

def remove_unescape(utterance:str):
    """Remove unescape chars from utterance 
    """ 
    utterance = re.sub('&gt;', '', utterance)
    utterance = re.sub('&lt;', '', utterance)
    utterance = re.sub('&amp;', '', utterance)
    return re.sub('#x200B;', '', utterance)
    
def clean_utterance(utterance:str, r=random):
    """Call all helper fns for cleaning an utterance
    """
    utterance = format_urls(utterance)
    utterance = remove_emojis(utterance)
    utterance = strip_bold(utterance)
    utterance = strip_italic(utterance)
    utterance = remove_unescape(utterance)

    # to randomize starting characterize capitalization
    try:
        utterance = r.choice([utterance[0].lower(),
                              utterance[0]])+utterance[1:]
    except IndexError:
        return ''

    return utterance

"""
Run detoxify model on Reddit Conversation Corpus (RCC) and output
filtered dataset compatible with RCC Corpus class based on tunable 
toxicity threshold. 
"""
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

def filter_corpus_toxicity(corpus:Corpus, id2results:dict[str], attribute2thresh:dict[str]={"toxicity": 0.9}):
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

def corpus_len(corpus:Corpus):
    return len([conv for conv in corpus.iter_conversations()])

def corpus_to_prompts(corpus:Corpus):
    convos = corpus.conversations.values()
    # we only keep the last five utterances (and also discard the front
    # because the front is the self-post on reddit)
    prompts = [[clean_utterance(j.text, R)
            for j in list(i.iter_utterances())
            if j.text.strip() != "[deleted]"
            and j.text.strip() != ""][1:][-2:]
           for i in convos]
    prompts = [[j for j in i if j.strip() != ""]
        for i in prompts]
    prompts = [i for i in prompts if len(i) != 0]
    return prompts

# download large conversation corpus
#corpus = Corpus(filename=download("reddit-corpus-small"))
#prompts = corpus_to_prompts(corpus)
#print(len(prompts))
#convo = corpus.random_conversation()
#print(filter_corpus_formatting(corpus))
#score_corpus(corpus, 'detox_results.jsonl')
#id2results = jsonl_to_dict('detox_results.jsonl')
#corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.9})

# Test formatting fixes

# # URLs
# url_test = '[hello](http://www.hello.com)'
# format_urls(url_test)
# url_test_multiple = '[hello](http://www.hello.com) filler filler filler [goodbye](http://www.goodbyye.com) lalala'
# format_urls(url_test_multiple)

# # Emojis and Unicode chars
# emoji_test = 'Only 8 months 🙄'
# print(remove_emojis(emoji_test))

# # Bold and italics
# bold_test = "**wow** that's **so** cool"
# print(strip_bold(bold_test))
# italic_test = "_wow_ that's __so__ cool"
# print(strip_italic(italic_test))

# # Escaped HTML tags
# unescape_test = "anon0: &gt;&gt; Completely blindsided &amp;#x200B;"
# print(remove_unescape(unescape_test))

# # Test all
# test = "**wow** check out [this website](http://www.website.com) &amp; it's __so__ cool 🙄"
# print(clean_utterance(test))
