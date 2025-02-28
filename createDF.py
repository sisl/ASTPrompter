# save conversation rollouts to a csv file
from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from environmentRoll2 import episode
import torch
import random
import pandas as pd
import numpy as np


def createCSVGPT():
    # adversarial model weights loaded
    #weights = "/home/allie11/ASTPrompter/models/BaseGPT2_nobp_newR_3t_best"
    #weights = "/home/allie11/ASTPrompter/models/NewMethod0.5_5t_44268S_best"
    #weights = "/home/allie11/ASTPrompter/models/gpt2New_.5_3H_best"
    weights = "/home/allie11/ASTPrompter/models/NewBaselineGPT_5t_LG_best"
    #weights = "/home/allie11/ASTPrompter/models/bigLlama0.5_lowerlr_best"
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16)
    adversary.tokenizer = AutoTokenizer.from_pretrained(weights)

    # standard, untouched GPT2 model as defender
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.bfloat16)
    defender.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    #defender.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)
    #defender.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    defender.model.eval()        

    # GPT 2 doesn't have a padding token, so we add it
    adversary.tokenizer.pad_token = adversary.tokenizer.eos_token
    defender.tokenizer.pad_token = defender.tokenizer.eos_token
    adversary.tokenizer.pad_token_id = adversary.tokenizer.eos_token_id
    defender.tokenizer.pad_token_id = defender.tokenizer.eos_token_id

    # load our initial corpus ahead of time
    corpus = Corpus(filename=download("reddit-corpus-small"))
    id2results = jsonl_to_dict('detox_results.jsonl')
    # only use prompts with toxicity scores less than 0.1 by deoxify
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)
    convos = list(corpus.conversations.values())
    
    # initialize dataframe
    column_names = ["prompt", "adv1", "def1", "adv2", "def2", "adv3", "def3", "adv4", "def4", "adv5", "def5", "adv6", "def6", "adv7", "def7", "adv8", "def8"]  
    df = pd.DataFrame(columns=column_names)

    # we only keep the last five utterances (and also discard the front
    # because the front is the self-post on reddit)
    prompts = [[clean_utterance(j.text)
                for j in list(i.iter_utterances())
                if j.text.strip() != "[deleted]"
                and j.text.strip() != ""][1:][-2:]
            for i in convos]
    prompts = [[j+" " for j in i if j.strip() != ""]
            for i in prompts]
    prompts = [i for i in prompts if len(i) != 0]
    #
    #print(len(prompts))
    #prompt2 = prompts[:100]
    #prompts = "Man Feels Like Whole Life is Just Endless Cycle of Eat, Sleep , Alien Abduction"
    #prompts = prompts[:2]
    #loop through all non-tox prompts
    for prompt in prompts:
        # print("==== PROMPT ====")
        # print(prompt)  # Now prompt is a single string, not a list
        
        convo = episode(adversary, defender, prompt, horizon=8, return_sequence=True)
        #print("".join("["+i+"] " for i in convo))
        
        convo = convo[len(prompt):]
        
        # get rid of newline characters becuase they lose anything in utterance after the newline
        convo = [utterance.strip() for utterance in convo]

        # Construct row with prompt as the first column
        full_row = [prompt] + convo

        # Create a DataFrame row
        row_df = pd.DataFrame([full_row], columns=column_names)

        # Add row to DataFrame
        df = pd.concat([df, row_df], ignore_index=True)
        
    df.to_csv("gpt.csv")
    
def createCSVLlama():
    # Load adversarial model (LLaMA with fine-tuned weights)
    weights = "/home/allie11/ASTPrompter/models/bigLlama0.5_lowerlr_best"
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16).cuda()
    adversary.tokenizer = AutoTokenizer.from_pretrained(weights)

    # Load defender model (Standard LLaMA-3.1-8B)
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        attn_implementation="flash_attention_2",  # Efficient attention
        torch_dtype=torch.bfloat16
    ).cuda()
    defender.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # Set padding tokens for LLaMA models
    adversary.tokenizer.pad_token = adversary.tokenizer.eos_token
    defender.tokenizer.pad_token = defender.tokenizer.eos_token

    # Load the initial corpus
    corpus = Corpus(filename=download("reddit-corpus-small"))
    id2results = jsonl_to_dict('detox_results.jsonl')

    # Filter out toxic and poorly formatted data
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)
    convos = list(corpus.conversations.values())

    # Initialize DataFrame
    column_names = ["prompt", "adv1", "def1", "adv2", "def2", "adv3", "def3", "adv4", "def4", "adv5", "def5", "adv6", "def6", "adv7", "def7", "adv8", "def8"]  
    df = pd.DataFrame(columns=column_names)

    # Extract the last two messages from each conversation (to use as a prompt)
    prompts = [[clean_utterance(j.text)
                for j in list(i.iter_utterances())
                if j.text.strip() not in ["[deleted]", ""]][1:][-2:]
               for i in convos]
    prompts = [[j + " " for j in i if j.strip() != ""] for i in prompts]
    prompts = [i for i in prompts if len(i) != 0]

    # Loop through non-toxic prompts
    for prompt in prompts:
        # print("==== PROMPT ====")
        # print(prompt)

        # Run conversation episodes
        convo = episode(adversary, defender, prompt, horizon=8, return_sequence=True)
        #print("".join(f"[{i}] " for i in convo))

        # Remove prompt portion from the conversation output
        convo = convo[len(prompt):]

        # Strip unnecessary newlines and whitespace
        convo = [utterance.strip() for utterance in convo]

        # Construct row with the prompt in the first column
        full_row = [prompt] + convo

        # Create a DataFrame row and append to main DataFrame
        row_df = pd.DataFrame([full_row], columns=column_names)
        df = pd.concat([df, row_df], ignore_index=True)

    # Save conversation trajectories
    df.to_csv("NewLlamaTs.csv", index=False)
    print("Saved to NewLlamaTs.csv")


createCSVLlama()