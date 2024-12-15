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
    def __init__(self, model="openai-community/gpt2", dont_init=False, model_load_params={}):

        # because huggingface accelerate / deepspeed may move the models
        # we lazily initialize where we actually are by looking it up (see
        # self.device) whenever we are asked. for now, we don't know
        self.__device = None

        if not dont_init:
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_load_params)
                                                        # attn_implementation="flash_attention_2",)
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def rollout(self, prompt, stop_sequence=None, temperature=0.7, top_p=0.7, do_sample=True, max_new_tokens=48, dont_stop=False, skip_special_tokens=False, **kwargs):
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
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        crit = None
        if stop_sequence:
            crit = EosListStoppingCriteria(stop_sequence)
        if isinstance(prompt, str):
            model_inputs = self.tokenizer([prompt], return_tensors="pt", 
                                          max_length=950, truncation=True).to(self.device)
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt", 
                                          max_length=950, truncation=True,
                                          padding=True).to(self.device)
        
        # if we are using DDP, the model sits in a wrapper object which we have
        # to untangle before generate
        underlying = self.model
        # if isinstance(underlying, DDP):
        #    underlying = self.model.module
        print("RO in")
        # we need to set 
        if stop_sequence:
            generated_ids = underlying.generate(**model_inputs, **kwargs, stopping_criteria = [crit],
                                                temperature=temperature, top_p=top_p,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                do_sample=do_sample, max_new_tokens=max_new_tokens)
        elif dont_stop:
            generated_ids = underlying.generate(**model_inputs, **kwargs,
                                                temperature=temperature, top_p=top_p,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                do_sample=do_sample, max_new_tokens=max_new_tokens, 
                                                stopping_criteria = [])
        else:
            generated_ids = underlying.generate(**model_inputs, **kwargs,
                                                temperature=temperature, top_p=top_p,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                do_sample=do_sample, max_new_tokens=max_new_tokens)

        print("RO out")

        if isinstance(prompt, str):
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)[0]
        else:
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

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
        x_enc = self.tokenizer([x], max_length=959, truncation=True)["input_ids"][0]
        y_enc = self.tokenizer([y], max_length=64, truncation=True)["input_ids"][0]
        model_inputs = torch.tensor([x_enc+y_enc]).to(device if device else self.device)
        underlying = self.model
        if isinstance(underlying, DDP):
            underlying = self.model.module

        res = self.model(input_ids=model_inputs)

        if isinstance(res, tuple):
            res = res[0].squeeze(0)
        else:
            res = res["logits"].squeeze(0)

        res = F.log_softmax(res, dim=1)

        # isolate the output components' probabilities; remember that
        # the last token omponent is one token beyond y (i.e. the final autoregression
        # token, beyond our value y, so we discard that)
        cond_log_probs = torch.gather(res[:len(x_enc)-1], 1, (torch.tensor(x_enc)[1:]).unsqueeze(1).to(device if device else self.device))
        all_log_probs = torch.gather(res[:-1], 1, (torch.tensor(x_enc+y_enc)[1:]).unsqueeze(1).to(device if device else self.device))

        log_probs = all_log_probs.sum(0) - cond_log_probs.sum(0)

        # sum of logs probs is product of probs
        return -log_probs[0]/len(y_enc)

    def logprob_batched(self, ys, device=None):
        """Obtain P(y) from the LM.

        In particular, we define P(y) as the sum of logprobs
        of each individual element.

        Parameters
        ----------
        y : str
            The entailments from the prompt to calculate the probability of.
        device : Optional[torch.device]
            The device to push the model inputs.

        Returns
        -------
        torch.Tensor: 1
            The probability of y.
        """

        # force a pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

        # combine the input and output and forward pass
        y_enc = self.tokenizer(ys, return_tensors="pt", padding=True, 
                truncation=True, max_length=1024).to(device if device else self.device)
        underlying = self.model
        if isinstance(underlying, DDP):
            underlying = self.model.module

        # we need to chop off the front because we don't
        # have the log prob of our first token
        loss_mask = (y_enc["attention_mask"][:,1:] != 0)

        # brrr
        res = self.model(**y_enc)["logits"]
        res = F.log_softmax(res, dim=2)

        # isolate the output components' probabilities; remember that
        # the last token omponent is one token beyond y (i.e. the final autoregression
        # token, beyond our value y, so we discard that)
        cond_log_probs = torch.gather(res[:,:-1,:], 2,
                                      (y_enc["input_ids"][:,1:]).unsqueeze(2).to(device if device else self.device)).squeeze(2)

        # sum of logs probs is product of probs
        # and we apply the mask to ignore logprob of padding
        return (cond_log_probs*loss_mask).sum(-1)

    def logprob(self, y, device=None):
        """Obtain P(y) from the LM.

        In particular, we define P(y) as the sum of logprobs
        of each individual element.

        Parameters
        ----------
        y : str
            The entailments from the prompt to calculate the probability of.
        device : Optional[torch.device]
            The device to push the model inputs.

        Returns
        -------
        torch.Tensor: 1
            The probability of y.
        """
        
        # combine the input and output and forward pass
        y_enc = self.tokenizer([y])["input_ids"][0]
        model_inputs = torch.tensor([y_enc]).to(device if device else self.device)
        underlying = self.model
        if isinstance(underlying, DDP):
            underlying = self.model.module

        res = self.model(input_ids=model_inputs)

        if isinstance(res, tuple):
            res = res[0].squeeze(0)
        else:
            res = res["logits"].squeeze(0)

        res = F.log_softmax(res, dim=1)

        # isolate the output components' probabilities; remember that
        # the last token omponent is one token beyond y (i.e. the final autoregression
        # token, beyond our value y, so we discard that)
        cond_log_probs = torch.gather(res[:-1], 1, (torch.tensor(y_enc)[1:]).unsqueeze(1).to(device if device else self.device))

        # sum of logs probs is product of probs
        return cond_log_probs.sum()

# lm = LanguageModel()

# lm.logprob_batched(["Hello! How's the weather?", "jerk", "beef jerky samurai!", "shibat shebat shemibu", "Advertisement"])

# lm.logprob("bjork there.")

# breakpoint()
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

