import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# to point to where we already have mistral downloaded
# TODO each cluster this needs to change for where huggingface
# put mistral
# for T5s, etc., we don't reall
# os.environ["HF_HOME"] = "/juice2/scr2/houjun/hf"

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

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
    def __init__(self, model="openai-community/gpt2", dont_init=False):

        # because huggingface accelerate / deepspeed may move the models
        # we lazily initialize where we actually are by looking it up (see
        # self.device) whenever we are asked. for now, we don't know
        self.__device = None

        if not dont_init:
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
                                                        # attn_implementation="flash_attention_2",)
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def to(self, device):
        self.model = self.model.to(device)

    def rollout(self, prompt, stop_sequence=None, temperature=0.7, top_p=0.7, do_sample=True, max_new_tokens=48, **kwargs):
        """Rollout our policy until a stop sequence.

        Parameters
        ----------
        prompt : str
            State to begin rollout at.
        stop_sequence : List[int]
            Stop sequence to stop rollout.
        **kwargs
            Rollout sampling parameters.
 
        Returns
        -------
        str
            Our rollout!
        """
        
        crit = None
        if stop_sequence:
            crit = EosListStoppingCriteria(stop_sequence)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        # if we are using DDP, the model sits in a wrapper object which we have
        # to untangle before generate
        underlying = self.model
        if isinstance(underlying, DDP):
            underlying = self.model.module
        if stop_sequence:
            generated_ids = underlying.generate(**model_inputs, **kwargs, stopping_criteria = [crit],
                                                temperature=temperature, top_p=top_p,
                                                do_sample=do_sample, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        else:
            generated_ids = underlying.generate(**model_inputs, **kwargs,
                                                temperature=temperature, top_p=top_p,
                                                do_sample=do_sample, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.batch_decode(generated_ids)[0]

    @property
    def device(self):
        if not self.__device:
            self.__device = next(self.model.parameters()).device

        return self.__device

    def perplexity(self, y, x="", device=None):
        """Obtain ppl(y|x) from the LM.

        REMEMBER: LOWER IS BETTER!

        In particular, we define P(y|x) as the sum of logprobs
        of each individual element.

        Parameters
        ----------
        x : str
           The prompt.
        y : str
            The entailments from the prompt to calculate the probability of.
        device : Optional[torch.device]
            The device to push the model inputs.

        Returns
        -------
        torch.Tensor: 1
            The perplexity of y given x
        """
        
        # combine the input and output and forward pass
        x_enc = self.tokenizer([x])["input_ids"][0]
        y_enc = self.tokenizer([y])["input_ids"][0]
        model_inputs = torch.tensor([x_enc+y_enc]).to(device if device else self.device)
        res = self.model(input_ids=model_inputs)["logits"].squeeze(0)
        res = F.log_softmax(res, dim=1)

        # isolate the output components' probabilities; remember that
        # the last token omponent is one token beyond y (i.e. the final autoregression
        # token, beyond our value y, so we discard that)
        log_probs = torch.gather(res[-len(y_enc)-1:-1], 1, (torch.tensor(y_enc)).unsqueeze(1).to(device if device else self.device))

        # sum of logs probs is product of probs
        return torch.exp(-log_probs.sum(0)[0]/len(y_enc))


# lm = LanguageModel()
# prompt1 = """
# anon1: what do you still fucking want?
# anon2: money and, you know, also
# anon1: """
# prompt2 = """
# anon1: that fucking person is so...
# anon2: """


# prompt = prompt1

# # get the anon stop token
# anon = lm.tokenizer("anon")["input_ids"][0]
# # rollout! yipee
# res = lm.rollout(prompt.strip(), stop_sequence=[anon])
# # get only the new generation
# new_utterance = res.replace(prompt, "").strip().split("\n")[0].strip()
# # as perplexity is the inverse of probability, we have to
# # negate it to measure likelyhood
# likelihood = -lm.perplexity(x=prompt, y=new_utterance)

# compute the entailment probabilities
# new_utterance

# get log probs of our rollout
# rollout_probs = lm.condition([res])
# rollout_probs.shape


# print()



# anon2: Here's an example of the bullshit that Kevin Sutherland was spouting on his"""

