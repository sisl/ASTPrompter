"""
reward.py
Reward model
"""

import torch
from perspective import call_perspective
from dxf import call_detoxify
from transformers import AutoModelForCausalLM, AutoTokenizer

min_window = 4
max_window = 12
eps = 1e-4
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
detox_engine = "detoxify"

text = ""
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
for win in range(min_window, min(max_window+1, len(toks)//2)):
    # to compute |t-j|
    middle = (win-1)/2
    for start in range(0, len(toks)-win+1, win):
        part = toks[start:start+win]
        weights = [(1/(abs(i-middle)+1)) for i in range(win)]
        # decode the string and call perspective AI
        part = tokenizer.decode(part)
        tox = detox(part)[part]["TOXICITY"]
        # scale toxicity score by weights
        weights = torch.tensor(weights)
        weights *= tox
        # add count (for averaging) as well as toxlevel
        tokenlevel_tox[start:start+win] += weights
        count[start:start+win] += 1
# perform averaging by element wise divide
tokenlevel_tox /= count
torch.nan_to_num(tokenlevel_tox)
count
# tokenizer.convert_ids_to_tokens(toks)
# tokenlevel_tox
    

# call_perspective("this person looks a little bit silly")
