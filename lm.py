import os
import torch

# to point to where we already have mistral downloaded
# TODO each cluster this needs to change for where huggingface
# put mistral
# for T5s, etc., we don't reall
os.environ["HF_HOME"] = "/juice2/scr2/houjun/hf"

from transformers import AutoModelForCausalLM, AutoTokenizer

# initialize models
class LanguageModel(object):
    """Language Model Inference

    Parameters
    ----------
    model: optional, str
        default = mistralai/Mistral-7B-v0.1
        The language model to use.

    Arguments
    ---------
    prompt: str
        The prompt to infer with the LM.
    **kwargs: inference parameters.
    """
    def __init__(self, model="mistralai/Mistral-7B-v0.1"):
        self.model = AutoModelForCausalLM.from_pretrained(model,
                                                    torch_dtype=torch.float16,
                                                    attn_implementation="flash_attention_2",
                                                    device_map="auto").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model)


    def __call__(self, prompt, temperature=1, top_p=0.9, do_sample=True, max_new_tokens=128, **kwargs):
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, **kwargs)
        return tokenizer.batch_decode(generated_ids)[0]


# toxicity prompt
# prompt = """
# anon1: What is an under eye skin graft?
# anon2: a poorly executed joke on my part. i don’t think people get skin grafts under their eyes, but watch Grey’s prove me wrong.
# anon1: No, that was good. You're funny and I 💚you.
# anon2: Here's an example of the bullshit that Kevin Sutherland was spouting on his"""

