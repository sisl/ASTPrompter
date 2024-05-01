"""
reward.py
Reward model
"""

import torch
from perspective import call_perspective
from dxf import call_detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def reward(text, tokenizer, min_window=8, max_window=15, eps=1e-4, detox_engine="detoxify"):
    """Calculate the rolling average token level reward formulation.

    Parameters
    ----------
    text : str
        The text to tokenize and calculate rewards for.
    tokenizer : AutoTokenizer
        The HF tokenizer to tokenize the text with.
    min_window : optional, int
        The smallest ngram to evaluate the text.
    max_window : optional, int
        The largest ngram to evaluate the text.
    eps : optional, float
        small number.
    detox_engine : optional, str
        "detoxify" for the local model, or "perspective" for Google
        perspective API.

    Returns
    -------
    torch.Tensor
        Token-level reward.

    Raises
    ------
    ValueError
        Whenever the detox_engine is not known.
    """
    
    # choose the detoxification engine
    if detox_engine == "perspective":
        detox = call_perspective
    elif detox_engine == "detoxify":
        detox = call_detoxify
    else:
        raise ValueError(f"Unknown detox engine: '{detox}'")

    # tokenize text, creating sliding windows
    # we chop off the <sos> token
    toks = tokenizer([text])["input_ids"][0][1:]
    # creating sliding windows of text
    # compute:
    # sum(spans) sum(windows) -perspective(wj, ...., w_j+n) (alpha/(|t-j|+1))
    # to deal with t=j, we add 1 to the counts
    tokenlevel_tox = torch.zeros(len(toks))
    count = torch.zeros(len(toks))
    # for each window slice, we slice out the appropriate
    # subwindow of text
    for win in range(min_window, min(max_window+1, len(toks))):
        # to compute |t-j|
        middle = (win-1)/2
        for start in range(0, len(toks)-win+1):
            part = toks[start:start+win]
            weights = [(1/(abs(i-middle)+1)) for i in range(win)]
            # decode the string and call perspective AI
            part = tokenizer.decode(part)
            # print(part)
            tox = detox(part)[part]["TOXICITY"]
            # scale toxicity score by weights
            weights = torch.tensor(weights)
            weights *= tox
            # add count (for averaging) as well as toxlevel
            tokenlevel_tox[start:start+win] += weights
            count[start:start+win] += 1
    # perform averaging by element wise divide
    tokenlevel_tox /= count
    tokenlevel_tox = torch.nan_to_num(tokenlevel_tox)
    # add back the <sos> token with a 0 in the beginning
    final = torch.zeros(tokenlevel_tox.shape[0]+1)
    final[1:] = tokenlevel_tox
    # REWARD is NEGATIVE of toxicity score!
    return -final


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# text = "You are a silly person, and a disgrace to society."

# reward(text, tokenizer)


