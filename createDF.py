# save conversation rollouts to a csv file
from transformers import AutoTokenizer, AutoModelForCausalLM
from convokit import Corpus, download, Conversation
from lm import LanguageModel
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance
import kagglehub
import os
from huggingface_hub import login
from environmentRoll2 import episode
import torch
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from llamaguard3 import moderate
import csv
import json


# Manually set Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = "garanguizdias"
os.environ['KAGGLE_KEY'] = "eebf167ece7ae27f36846fa0bda2e333"

# Function to download and load headlines
def download_news_dataset():
    dataset_path = kagglehub.dataset_download("rmisra/news-category-dataset")

    # Find the extracted JSON file
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                return os.path.join(root, file)

    raise FileNotFoundError("Dataset JSON file not found in the extracted folder.")

def load_headlines(num_samples=1000):
    try:
        # Download and get the dataset path
        dataset_file = download_news_dataset()

        # Load JSON dataset
        df = pd.read_json(dataset_file, lines=True)

        # Extract headlines
        headlines = df["headline"].dropna().tolist()

        # randomly select num_samples headlines
        random.shuffle(headlines)
        # Select a subset of headlines (adjust as needed)
        headlines = headlines[:num_samples]
        headlines = [i.strip() for i in headlines if len(i.strip()) > 0]
        
        # make sure every value in headlines is a list
        headlines = [[i] for i in headlines]
        return headlines

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_wiki_articles(num_samples=2):
    # implort wikipedia dataset from kaggle - english 2022
    data = load_dataset("wikipedia", "20220301.en")
    # each example in dataset looks like this: {'id': '1',
        # 'url': 'https://simple.wikipedia.org/wiki/April',
        # 'title': 'April',
        # 'text': 'April is the fourth month...'
        # }
        
    # Select the "train" split before slicing
    train_data = data['train']
    # Randomly select num_samples indices from the dataset
    total_size = len(train_data)
    random_indices = random.sample(range(total_size), num_samples)
    
    # Select random samples
    small_data = train_data.select(random_indices)
    
    # List to store extracted text as prompts
    promptsWiki = []

    # Process each article
    for i in range(num_samples):
        text = small_data[i]['text']
        
        # Extract the first two sentences
        sentences = sent_tokenize(text)  # Tokenize into sentences
        first_two_sentences = " ".join(sentences[:2])  # Join the first two
        #print(first_two_sentences)
        
        # Append the extracted text to prompts list
        promptsWiki.append(first_two_sentences)
    # make sure each value in promptsWiki is a list
    promptsWiki = [[i] for i in promptsWiki]
    return promptsWiki  # Return the list of prompts
    
def load_coding_questions(num_samples=2):
    ds = load_dataset("jtatman/python-code-dataset-500k")
    prompts = []
    # Select the "train" split before slicing
    train_data = ds['train']
    # Randomly select num_samples indices from the dataset
    total_size = len(train_data)
    random_indices = random.sample(range(total_size), num_samples)
    small_data = train_data.select(random_indices)
    # print one example
    #print(small_data[0]["instruction"])
    for i in range(num_samples):
        prompts.append(small_data[i]["instruction"])
    
    # make sure each value in prompts is a list
    prompts = [[i] for i in prompts]
    return prompts
      
def createCSVGPT(weights, defenderWeights):
    # if non-hardened defender model: defenderWeights = "openai-community/gpt2"
    # adversarial model weights loaded
    #weights = "/home/allie11/ASTPrompter/models/BaseGPT2_nobp_newR_3t_best"
    #weights = "/home/allie11/ASTPrompter/models/NewMethod0.5_5t_44268S_best"
    #weights = "/home/allie11/ASTPrompter/models/gpt2New_.5_3H_best"
    #weights = "/home/allie11/ASTPrompter/models/NewBaselineGPT_5t_LG_best"
    #weights = "/home/allie11/ASTPrompter/models/bigLlama0.5_lowerlr_best"
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16)
    adversary.tokenizer = AutoTokenizer.from_pretrained(weights)

    # standard, untouched GPT2 model as defender
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained(defenderWeights, torch_dtype=torch.bfloat16)
    defender.tokenizer = AutoTokenizer.from_pretrained(defenderWeights)
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
    prompts = load_headlines(num_samples=1100)
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
        
    df.to_csv("Baseline3t_v_HardGPT2_trajectories.csv")
    
def createCSVLlama(weights, news=False):
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16).cuda()
    adversary.tokenizer = AutoTokenizer.from_pretrained(weights)

    # load defender model (Standard LLaMA-3.1-8B)
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16
    ).cuda()
    defender.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # set padding tokens for LLaMA models - dif than GPT2
    adversary.tokenizer.pad_token = adversary.tokenizer.eos_token
    defender.tokenizer.pad_token = defender.tokenizer.eos_token

    if news==False: 
        corpus = Corpus(filename=download("reddit-corpus-small"))
        id2results = jsonl_to_dict('detox_results.jsonl')

        # filter out toxic and poorly formatted data
        corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
        corpus = filter_corpus_formatting(corpus)
        convos = list(corpus.conversations.values())
        # extract the last two messages from each conversation (to use as a prompt)
        prompts = [[clean_utterance(j.text)
                    for j in list(i.iter_utterances())
                    if j.text.strip() not in ["[deleted]", ""]][1:][-2:]
                for i in convos]
        prompts = [[j + " " for j in i if j.strip() != ""] for i in prompts]
        prompts = [i for i in prompts if len(i) != 0]

    if news==True:
        #trying with news headlines
        prompts = load_headlines(num_samples=1100)
        # Filter out empty prompts
        prompts = [i.strip() for i in prompts if len(i.strip()) > 0]
        prompts = [[i] for i in prompts]
        prompts = prompts[:10]
        #print(prompts)
        
        
    # Initialize DataFrame
    column_names = ["prompt", "adv1", "def1", "adv2", "def2", "adv3", "def3", "adv4", "def4", "adv5", "def5", "adv6", "def6", "adv7", "def7", "adv8", "def8"]  
    df = pd.DataFrame(columns=column_names)
    #df2 = pd.DataFrame(columns=column_names)


    # loop through non-toxic prompts
    for prompt in prompts:
        print("==== PROMPT ====")
        print(prompt)

        # run conversation episodes
        convo = episode(adversary, defender, prompt, horizon=8, return_sequence=True)
        print(convo)
        print("".join(f"[{i}] " for i in convo))

        # remove prompt portion from the conversation output
        convo = convo[len(prompt):]

        # strip unnecessary newlines and whitespace
        convo = [utterance.strip() for utterance in convo]
        #print(len(convo))

        # construct row with the prompt in the first column
        full_row = [prompt] + convo

        # create a DataFrame row and append to main DataFrame
        row_df = pd.DataFrame([full_row], columns=column_names)
        df = pd.concat([df, row_df], ignore_index=True)

    # Save conversation trajectories
    df.to_csv("LlamaBaselineLg.csv", index=False)
    #df2.to_csv("test2LLamaNEWDF.csv", index=False)
    print("Saved to LlamaBaselineLg.csv")


def createMetaDFLlama(weights, num_samples=300):
    # load adversary
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(
        weights, torch_dtype=torch.bfloat16
    ).cuda()
    adversary.tokenizer = AutoTokenizer.from_pretrained(weights)

    # load defender
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).cuda()
    defender.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    adversary.tokenizer.pad_token = adversary.tokenizer.eos_token
    defender.tokenizer.pad_token = defender.tokenizer.eos_token

    def process_prompts(prompt_type, prompts, adversary, defender, horizon=8):
        rows = []
        for prompt in prompts:
            # join the prompt utterances into one string and remove newlines
            prompt_str = " ".join(prompt).replace("\n", " ").strip()
            
            # run conversation episode with the given prompt.
            convo = episode(adversary, defender, prompt, horizon=horizon, return_sequence=True)
            # remove the prompt portion
            convo = convo[len(prompt):]
            # replace newlines in each utterance 
            convo = [utterance.replace("\n", " ").strip() for utterance in convo]
            # convert the convo list to a JSON string 
            convo_str = json.dumps(convo)
            
            toxList = []
            # process each utterance moderation output.
            for i in range(len(convo)):
                toxScore, labelList = moderate(convo[i])
                if i % 2 == 0:
                    # adv utt = labelList
                    if labelList and len(labelList) > 0:
                        cleaned_label = labelList[0].replace("\n", " ").strip()
                    else:
                        cleaned_label = ""
                    toxList.append(cleaned_label)
                else:
                    # for defender utterances, save toxicity score.
                    toxList.append(toxScore)
            
            # row: [prompt type, prompt, convo, adv1, def1, adv2, def2, ... adv8, def8]
            row = [prompt_type, prompt_str, convo_str] + toxList
            
            if len(row) == 19:
                rows.append(row)
        return rows

    
    #  Reddit prompts: load from corpus
    corpus = Corpus(filename=download("reddit-corpus-small"))
    id2results = jsonl_to_dict('detox_results.jsonl')
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)
    convos1 = list(corpus.conversations.values())
    promptsRed = [
        [clean_utterance(j.text)
         for j in list(i.iter_utterances())
         if j.text.strip() not in ["[deleted]", ""]][1:][-2:]
        for i in convos1
    ]
    promptsRed = [[j + " " for j in prompt if j.strip() != ""] for prompt in promptsRed]
    promptsRed = [p for p in promptsRed if len(p) != 0][:num_samples]

    #News prompts
    promptsNews = load_headlines(num_samples)
    
    # Wikipedia article prompts
    promptsWiki = load_wiki_articles(num_samples)
    
    # Coding question prompts
    promptsCode = load_coding_questions(num_samples)
    
    all_rows = []
    all_rows.extend(process_prompts("Reddit", promptsRed, adversary, defender, horizon=8))
    all_rows.extend(process_prompts("News", promptsNews, adversary, defender, horizon=8))
    all_rows.extend(process_prompts("Wiki", promptsWiki, adversary, defender, horizon=8))
    all_rows.extend(process_prompts("Code", promptsCode, adversary, defender, horizon=8))

    # define df cols
    column_names = [
        "prompt type", "prompt", "convo",
        "adv1 tox class", "def1 tox score",
        "adv2 tox class", "def2 tox score",
        "adv3 tox class", "def3 tox score",
        "adv4 tox class", "def4 tox score",
        "adv5 tox class", "def5 tox score",
        "adv6 tox class", "def6 tox score",
        "adv7 tox class", "def7 tox score",
        "adv8 tox class", "def8 tox score"
    ]

    df = pd.DataFrame(all_rows, columns=column_names)

    df.to_csv("LlamaBaselineDiversePromptsLarge.csv", index=False, quoting=csv.QUOTE_ALL)
    print("Saved to LlamaDiversePromptsLarge.csv")
 
if __name__ =="__main__":
    weights = "/home/allie11/ASTPrompter/models/llama_v_llama_og_best"
    # call function - give model weights as input
    # running on coolio tmux 
    #createCSVGPT(weights = "/home/allie11/ASTPrompter/models/gpt2New_.5_3H_best", defenderWeights = "/home/allie11/ASTPrompter/models/gpt2_hardened_best")
    # running on spring tmux
    #createCSVGPT(weights = "/home/allie11/ASTPrompter/models/BaseGPT2_nobp_newR_3t_best", defenderWeights = "/home/allie11/ASTPrompter/models/gpt2_hardened_best")
    #createCSVLlama(weights, news=False)
    #(num_samples=10)
    #load_coding_questions(num_samples=2)
    createMetaDFLlama(weights)