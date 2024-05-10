import torch
from detoxify import Detoxify

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
MODEL = Detoxify('original', device=DEVICE)

def call_detoxify(text):
    res = MODEL.predict(text)
    return res




