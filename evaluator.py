from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from lm import *
from peft import LoraConfig
from environment import *
import torch
from torch.nn.utils import clip_grad_norm_
import os
import wandb
import pickle
import json

logger = get_logger("ast")

class Evaluator:

    def __init__(self, args, **kwargs):

        # cache horizon
        self.horizon = args.horizon

        # initialize early the accelator
        self.accelerator = Accelerator(**kwargs.get("accelerator_kwargs", {}))

        # because the PPO wrapper chops the end off and add
        # a value head, we can't just naively initalize a GPT-2
        # supposedly, though, APIs are the same so we can
        # just use it in our inference wrapper

        self.adversary = LanguageModel(dont_init=True)
        self.adversary.model = AutoModelForCausalLM.from_pretrained(args.weights)
        self.adversary.tokenizer = AutoTokenizer.from_pretrained(args.weights)

        self.defender = LanguageModel(dont_init=True)
        self.defender.model = AutoModelForCausalLM.from_pretrained(args.defense)
        self.defender.tokenizer = AutoTokenizer.from_pretrained(args.defense)
        self.defender.model.eval()

        # GPT 2 doesn't have a padding token, so we add it
        self.adversary.tokenizer.pad_token = self.adversary.tokenizer.eos_token
        self.defender.tokenizer.pad_token = self.defender.tokenizer.eos_token
        self.adversary.tokenizer.pad_token_id = self.adversary.tokenizer.eos_token_id
        self.defender.tokenizer.pad_token_id = self.defender.tokenizer.eos_token_id

        # because the accelerator may move models to weird places, we 
        # account for that
        (self.adversary.model,
         self.defender.model) = self.accelerator.prepare(self.adversary.model,
                                                         self.defender.model)

        self.args = args

        save_name = f"eval_{args.weights.split('/')[-1]}"
        if args.save_name:
            save_name = args.save_name
        self.save_dir = os.path.join(args.save_dir, save_name)
        os.makedirs(self.save_dir, exists_ok=True)
        self.args = args
        self.__results_cache = []

    def load(self, data):
        """Make a distributed dataset from stings for training.

        Parameters
        ----------
        steps : List[List[str]]
            Prompt strings.

        Returns
        -------
        torch.utils.data.DataLoader
            The dataloader to pass to self.epoch.
        """

        class TrainerDataset(Dataset):
            def __init__(self, data):
                super().__init__()
                self.__data = data
            def __getitem__(self, x):
                return self.__data[x]
            def __len__(self):
                return len(self.__data)

        ds = TrainerDataset(steps)
        # batch_size = 1 because we will blow each batch
        # up to an entire dialogue
        dl = DataLoader(ds, 1) 

        # huggingface accelerate may ship the dataset
        # off to different processes, etc.
        return self.accelerator.prepare(dl)

    def __call__(self, dl, log_every=10):
        for indx, i in enumerate(tqdm(iter(dl), total=len(dl))):
            # we do i[0] because each is in a batch of 1
            self.__results_cache += episode(self.adversary, self.defender, i[0], self.horizon)
            if indx % log_every == 0:
                self.report()

    def report(self):
        rewards = [i.reward_w for i in self.__results_cache]
        ast_toxes = [i.reward_component.ast_tox for i in self.__results_cache]
        combined_toxes = [i.reward_component.combined_toxes for i in self.__results_cache]
        ast_ppls = [i.reward_component.ast_ppls for i in self.__results_cache]

        result = {
            "rewards": {
                "max": max(rewards),
                "min": min(rewards),
                "mean": sum(rewards)/len(rewards),
            },
            "tox": {
                "ast": {
                    "max": max(ast_toxes),
                    "min": min(ast_toxes),
                    "mean": sum(ast_toxes)/len(ast_toxes),
                },
                "combined": {
                    "max": max(combined_toxes),
                    "min": min(combined_toxes),
                    "mean": sum(combined_toxes)/len(combined_toxes),
                }
            },
            "ppl": {
                "ast": {
                    "max": max(ast_ppls),
                    "min": min(ast_ppls),
                    "mean": sum(ast_ppls)/len(ast_ppls),
                }
            },
            "meta": {
                "count": len(self.__results_cache),
                "adversary": self.args.weights,
                "defender": self.args.defense
            }
        }

        with open(os.path.join(self.save_dir, "results.json"), 'w') as df:
            json.dump(result, df, indent=4)
        with open(os.path.join(self.save_dir, "results.pkl"), 'wb') as df:
            pickle.dump(self.__results_cache, df)
        
