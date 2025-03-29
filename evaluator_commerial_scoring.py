import jsonlines
from llamaguard3 import moderate_
from toxicity.dxf import call_detoxify
from tqdm import tqdm

with jsonlines.open("./results/llama_v_claude_traces.jsonl") as reader:
    docs = [i for i in reader]

scores = []
for i in tqdm(docs):
    scores.append(moderate_(i[1:])[0])

print(sum(scores)/len(scores)) # 0.01999516036889072

scores_dtf = []
for i in tqdm(docs):
    scores_dtf.append(call_detoxify(" ".join(j["content"] for j in i[1:]))["toxicity"])

print(sum(scores_dtf)/len(scores_dtf)) # 0.022398287

import torch
torch.tensor(scores).topk(k=10)
docs[28]


