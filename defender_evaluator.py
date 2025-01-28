from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from statistics import stdev, mean
from accelerate.utils.tqdm import tqdm
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from lm import *
from peft import LoraConfig
import torch
from torch.nn.utils import clip_grad_norm_
import os
import wandb
import pickle
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import pandas as pd
import argparse
import copy
from toxicity.dxf import call_detoxify
from environment import reward, ASTStep, RewardComponent

from datasets import load_dataset

logger = get_logger("ast")

def get_text_from_conversation(conversation):
    return "".join([i['text'] for i in conversation])

class BadEvaluator:

    def __init__(self, args, **kwargs):

        # cache horizon and k
        # self.horizon = args.horizon
        self.k = args.k
        self.turn_order = args.turn_order

        # Save rollout params
        self.max_new_tokens = args.max_new_tokens
        self.min_new_tokens = args.min_new_tokens
        self.repetition_penalty = args.repetition_penalty
        self.temperature = args.temperature

        # initialize early the accelator
        self.accelerator = Accelerator(**kwargs.get("accelerator_kwargs", {}))

        self.attacker = LanguageModel(dont_init=True)
        self.attacker.model = AutoModelForCausalLM.from_pretrained(args.attacker)
        if 'tiny' in args.attacker:
            # Dirty hack to get around the fact that the tokenizer is not in the model directory
            self.attacker.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', padding_side='left', max_length=512)
        else:
            self.attacker.tokenizer = AutoTokenizer.from_pretrained(args.attacker, padding_side='left')
        self.attacker.model.eval()

        self.defender = LanguageModel(dont_init=True)
        self.defender.model = AutoModelForCausalLM.from_pretrained(args.defender)
        self.defender.tokenizer = AutoTokenizer.from_pretrained(args.defender, padding_side='left')
        self.defender.model.eval()

        self.baseline = LanguageModel(dont_init=True)
        self.baseline.model = AutoModelForCausalLM.from_pretrained(args.baseline)
        self.baseline.tokenizer = AutoTokenizer.from_pretrained(args.baseline, padding_side='left')
        self.baseline.model.eval()

        # Hack to force evaluation on second GPU to accelerate evals
        # For somereason prepare isn't doing this automatically
        # device = 'cuda:1'
        device = self.accelerator.device

        # Force models to GPU just in case
        print("Accelerator device: ", device)
        self.attacker.model.to(device)
        self.defender.model.to(device)
        self.baseline.model.to(device)

        # GPT 2 doesn't have a padding token, so we add it
        if 'gpt2' in args.attacker:
            self.attacker.tokenizer.pad_token = self.attacker.tokenizer.eos_token
            self.attacker.tokenizer.pad_token_id = self.attacker.tokenizer.eos_token_id

        if 'gpt2' in args.defender:
            self.defender.tokenizer.pad_token = self.defender.tokenizer.eos_token
            self.defender.tokenizer.pad_token_id = self.defender.tokenizer.eos_token_id

        if 'gpt2' in args.baseline:
            self.baseline.tokenizer.pad_token = self.baseline.tokenizer.eos_token
            self.baseline.tokenizer.pad_token_id = self.baseline.tokenizer.eos_token_id

        # because the accelerator may move models to weird places, we 
        # account for that
        # self.attacker.model, self.defender.model, self.baseline.model = self.accelerator.prepare(self.attacker.model, self.defender.model, self.baseline.model)
        
        self.args = args
       
        save_name = f"eval_bad"  # Default save location
        if args.save_name:
            save_name = args.save_name
        self.save_dir = os.path.join(args.save_dir, save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.args = args
        self.__results_cache = []

    def ds_to_df(self):
        """
        Convert tfds Dataset -> pandas DataFrame to allow filtering the data
        """
        ds = tfds.load('bot_adversarial_dialogue/dialogue_datasets', split='train', shuffle_files=False, download=True)
        # This is the suff we care about
        speakers = []
        texts = []
        labels = []
        dialogue_ids = []
        round_ids = []

        for idx, example in enumerate(list(ds)):
            speakers.append(example["speaker_to_eval"].numpy())
            texts.append(str(example["text"].numpy()))
            labels.append(example["labels"].numpy())
            dialogue_ids.append(example["dialogue_id"].numpy())
            round_ids.append(example["round_id"].numpy())


        data = {"speaker": speakers, "text": texts, "label": labels,
                "dialogue_id": dialogue_ids, "round_id": round_ids}

        return pd.DataFrame(data)

    def load_bad_prompts(self, seed=24, horizon=3):
        """
        Get the data ready for eval!

        1. tsds Datset -> pd DataFrame
        2. filter df: we only care about user utterances from converations 
        that are not ok (these will be the toxic ones)
        3. group by dialogue_id and select the last horizon turns 
        (this is where the toxicity happens)
        4. select self.k conversations to evaluate on

        Parameters
        ----------
        seed : int (default 24)
            random seed for reproducibility when we select our conversations
        
        horizon : int (default 3)
            we will take the last horizon turns of a conversation

        Returns
        --------
        list[list[str]]
            the conversations to eval on in the form
            [['turn1', 'turn2', 'turn3'], ['turn1', 'turn2', 'turn3']]
        """
        # Load to df
        df = self.ds_to_df()
        human_df = df[df["speaker"] == b'human']
        bad_human_df = human_df[human_df["label"] == 1]
        
        # Get the last horizon turns
        grouped = bad_human_df.sort_values(by="round_id").groupby("dialogue_id", as_index=False).apply(lambda x: x.tail(horizon)).reset_index()
        id_to_turns = {}
        for idx, row in grouped.iterrows():
            dialogue_id = row["dialogue_id"]
            text = row["text"]
            if dialogue_id not in id_to_turns.keys():
                id_to_turns[dialogue_id] = [text]
            else:
                id_to_turns[dialogue_id].append(text)
        
        # Remove conversations with too few turns
        drop_keys = []
        for id in id_to_turns.keys():
            if len(id_to_turns[id]) < horizon:
                drop_keys.append(id)
        for key in drop_keys:
            id_to_turns.pop(key)

        # Get examples for eval
        if self.k > 0:
            keys = list(id_to_turns.keys())
            R = random.Random(seed)
            selected_ids = R.choices(keys, k=self.k)
            bad_prompts = []
            for id in selected_ids:
                turns = id_to_turns[id]
                bad_prompts.append(turns)
        else:
            bad_prompts = list(id_to_turns.values())

        # Merge prompts into a single string remove incorrect byte string formmating artifacts
        # bad_prompts = [' '.join(turn.removeprefix('"b\'').removeprefix('\'b"').removesuffix('\'"').removesuffix('"\'')) for prompt in bad_prompts for turn in prompt]

        # Reprocess prompts to remove byte string formatting artifacts
        prompts = []

        for raw_prompt in bad_prompts:
            prompt = []
            for p in raw_prompt:
                prompt.append(p.removeprefix('b\'').removeprefix('b"').removesuffix('\'').removesuffix('"'))

            prompts.append(' '.join(prompt))
    
        return prompts
    
    def load_rtp_prompts(self, seed=24):

        ds = load_dataset("allenai/real-toxicity-prompts")

        # get a list of the text of the prompts
        rtp_prompts = [prompt['text'] for prompt in ds['train']['prompt']]

        if self.k > 0:
            R = random.Random(seed)
            rtp_prompts = R.choices(rtp_prompts, k=self.k)

        return rtp_prompts

    def _generate_rollout(self, attacker, defender, prompt):

        # Copy the promp
        text = copy.deepcopy(prompt)

        conversation = [{
            "text": copy.deepcopy(prompt),
            "role": "prompt",
            "model": "prompt",
            "toxicity": float(call_detoxify(text)["toxicity"])
        }]


        for move in self.turn_order:
            if move == "atk":
                # Generate a continuation of the text using the attacker model
                new_text = attacker.rollout(text, max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty, temperature=self.temperature, skip_special_tokens=True, min_new_tokens=self.min_new_tokens)
                
                # Extract just the new utterance
                new_utterance = new_text.replace(text, '')

                conversation.append({
                    "text": new_utterance,
                    "role": "atk",
                    "toxicity": float(call_detoxify(new_utterance)["toxicity"])
                })

                # Update the text with the new utterance
                text = new_text
            
            
            elif move == "def":
                # Generate a continuation of the text using the defender model
                new_text = defender.rollout(text, max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty, temperature=self.temperature, skip_special_tokens=True, min_new_tokens=self.min_new_tokens)
                
                # Extract just the new utterance
                new_utterance = new_text.replace(text, '')

                conversation.append({
                    "text": new_utterance,
                    "role": "def",
                    "toxicity": float(call_detoxify(new_utterance)["toxicity"])
                })

                # Update the text with the new utterance
                text = new_text
            
            else:
                raise ValueError(f"Invalid move: {move}")
            
        return conversation

    def episode(self, prompt):
        """Perform a single episode of the environment.

        Parameters
        ----------
        ast_prompts : list[str]
            set of prompts to evaluate on for a single episode
            form ['turn1', 'turn2', 'turn3']
        
        defender : LanguageModel
            The defender model responding.
        
        depth : int
            keep track of which turn we're on. current ast_ut is
            ast_prompmts[depth]

        prompt_src : List[str]
            The prompt set to start with. starts as [""] and then
            we add subsequent turns

        horizon : int
            length of the horizon (number of turns)

        Returns
        -------
        List[ASTStep], List[float]
            Steps, Rewards.
        """

        # Generate two rollouts first expected to be toxic, second expected to be nontoxic
        baseline_conversation = self._generate_rollout(self.attacker, self.baseline, prompt)
        defender_convseration = self._generate_rollout(self.attacker, self.defender, prompt)

        # Score each of the rollouts
        baseline_toxicity = float(call_detoxify(get_text_from_conversation(baseline_conversation))["toxicity"])
        defender_toxicity = float(call_detoxify(get_text_from_conversation(defender_convseration))["toxicity"])

        result = {
            "baseline_conversation": baseline_conversation,
            "defender_conversation": defender_convseration,
            "baseline_toxicity": baseline_toxicity,
            "defender_toxicity": defender_toxicity
        }

        return result
    
    def __call__(self, prompts, log_every=10):
        for indx, prompt in enumerate(tqdm(prompts, total=len(prompts))):
            self.__results_cache.append(self.episode(prompt))
            
            if indx % log_every == 0 and indx > 0:
                self.report()

    def report(self):
        baseline_attacker_toxicity = []
        baseline_defender_toxicity = []
        baseline_conversation_toxicity = []

        defender_attacker_toxicity = []
        defender_defender_toxicity = []
        defender_conversation_toxicity = []

        # Process baseline conversations
        for rollout in self.__results_cache:
            # Process baseline conversation of rollout
            baseline_conversation_toxicity.append(rollout["baseline_toxicity"])
            for utterance in rollout["baseline_conversation"]:
                if utterance["role"] == "atk":
                    baseline_attacker_toxicity.append(utterance["toxicity"])
                elif utterance["role"] == "def":
                    baseline_defender_toxicity.append(utterance["toxicity"])
            
            # Process defender conversation of rollout
            defender_conversation_toxicity.append(rollout["defender_toxicity"])
            for utterance in rollout["defender_conversation"]:
                if utterance["role"] == "atk":
                    defender_attacker_toxicity.append(utterance["toxicity"])
                elif utterance["role"] == "def":
                    defender_defender_toxicity.append(utterance["toxicity"])

        result = {
            "baseline": {
                "attacker_utterances": {
                    "toxicity_mean": mean(baseline_attacker_toxicity),
                    "toxicity_sdev": stdev(baseline_attacker_toxicity),
                    "toxicity_max": max(baseline_attacker_toxicity),
                    "toxicity_min": min(baseline_attacker_toxicity),
                },
                "defender_utterances": {
                    "toxicity_mean": mean(baseline_defender_toxicity),
                    "toxicity_sdev": stdev(baseline_defender_toxicity),
                    "toxicity_max": max(baseline_defender_toxicity),
                    "toxicity_min": min(baseline_defender_toxicity),
                },
                "conversation": {
                    "toxicity_mean": mean(baseline_conversation_toxicity),
                    "toxicity_sdev": stdev(baseline_conversation_toxicity),
                    "toxicity_max": max(baseline_conversation_toxicity),
                    "toxicity_min": min(baseline_conversation_toxicity),
                }
            },
            "hardended_defender": {
                "attacker_utterances": {
                    "toxicity_mean": mean(defender_attacker_toxicity),
                    "toxicity_sdev": stdev(defender_attacker_toxicity),
                    "toxicity_max": max(defender_attacker_toxicity),
                    "toxicity_min": min(defender_attacker_toxicity),
                },
                "defender_utterances": {
                    "toxicity_mean": mean(defender_defender_toxicity),
                    "toxicity_sdev": stdev(defender_defender_toxicity),
                    "toxicity_max": max(defender_defender_toxicity),
                    "toxicity_min": min(defender_defender_toxicity),
                },
                "conversation": {
                    "toxicity_mean": mean(defender_conversation_toxicity),
                    "toxicity_sdev": stdev(defender_conversation_toxicity),
                    "toxicity_max": max(defender_conversation_toxicity),
                    "toxicity_min": min(defender_conversation_toxicity),
                }
            },
            "meta": {
                "count": len(self.__results_cache),
                "dataset": self.args.dataset,
                "defender": self.args.defender,
                "attacker": self.args.attacker,
                "baseline": self.args.baseline,
                "turn_order": self.turn_order,
                "max_new_tokens": self.max_new_tokens,
                "min_new_tokens": self.min_new_tokens,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature
            }
        }

        with open(os.path.join(self.save_dir, "results.json"), 'w') as df:
            json.dump(result, df, indent=4)
        with open(os.path.join(self.save_dir, "results.pkl"), 'wb') as df:
            pickle.dump(self.__results_cache, df)

        if self.args.save_rollouts:
            with open(os.path.join(self.save_dir, "rollouts.pkl"), 'wb') as df:
                pickle.dump(self.__results_cache, df)

            with open(os.path.join(self.save_dir, "rollouts.json"), 'w') as df:
                json.dump(self.__results_cache, df, indent=4)

if __name__ == "__main__":
    # set random seed for reproducability
    set_seed(24)

    # establish the arguments of this system
    parser = argparse.ArgumentParser(description='BAD AST Evaluator')
    parser.add_argument('--attacker', type=str, help='what weights should we use for attacker?',
                        default="openai-community/gpt2")
    parser.add_argument('--defender', type=str, help='what weights should we use for defense?',
                        default="openai-community/gpt2")
    parser.add_argument('--baseline', type=str, help='what weights should we use for baseline?',
                        default="openai-community/gpt2")
    # parser.add_argument('--horizon', type=int, help='how many turns to self-play?',
    #                     default=3)
    parser.add_argument('--save-dir', type=str, default='results',
                        help='prefix of the model save dir, default "results"')
    parser.add_argument('--save-name', type=str, default=None,
                        help='what to name the results')
    parser.add_argument('--k', type=int, default=0,
                        help='how many examples to eval')
    parser.add_argument('--dataset', type=str, default='bad')
    parser.add_argument('--turn-order', nargs='+', default=['atk', 'def', 'atk', 'def', 'atk', 'def'])
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--min-new-tokens', type=int, default=2)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--save-rollouts', action='store_true')

    args = parser.parse_args()

    print(f'Attacker Model: {args.attacker}')
    print(f'Defender Model: {args.defender}')
    print(f'Baseline Model: {args.baseline}')
    print(f"Evaluating {args.k} prompts") if args.k > 0 else print("Evaluating all prompts")
    print(f'Using turn order: {args.turn_order}')

    evaluator = BadEvaluator(args)
    
    print('Loading prompts...')
    if args.dataset == 'bad':
        prompts = evaluator.load_bad_prompts()
    elif args.dataset == 'rtp':
        prompts = evaluator.load_rtp_prompts()

    print(f'Beginning evaluation on {len(prompts)} prompts...')
    evaluator(prompts)
