import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer # Had to install trl from source to get this to run
import json
import random
from accelerate import Accelerator
import torch
from lm import LanguageModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import os
import wandb

os.environ["WANDB_PROJECT"] = "sft"
accelerator = Accelerator(project_dir="sft_out/", log_with="wandb")

R = random.Random(24)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

transformers.logging.set_verbosity_info()

# Train on toxic comments
dataset = load_dataset("csv", data_files={"train":"toxic_comments.csv"})

def formatting_prompts_func(example):
    formatted_texts = []

    for text in example["text"]:
        formatted_texts.append(text)

    return formatted_texts

def preprocess_function(example, tokenizer):
    x = tokenizer.encode(example["text"], add_special_tokens=True, max_length=1024, truncation=True)
    #x = tokenizer(example["text"])
    return {"input_ids": x}

# Eval on RTP (for now)
with open("prompts.jsonl", 'r') as df:
    lines = df.readlines()
    data = json.loads("["+",".join(lines)+"]")
    prompts_rtp = [(R.choice([i["prompt"]["text"][0].lower(),
        i["prompt"]["text"][0]])+i["prompt"]["text"][1:], 
        R.choice([i["continuation"]["text"][0].lower(),
            i["continuation"]["text"][0]])+i["continuation"]["text"][1:])
        for i in data if i["continuation"]["toxicity"]
        and i["continuation"]["toxicity"] > 0.3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFT Trainer')
    parser.add_argument('--weights', type=str, help='which model shall we evaluate?', default="openai-community/gpt2")
    parser.add_argument('--defense', type=str, help='what weights should we use for defense?',
                        default="openai-community/gpt2")
    parser.add_argument('--warmup_steps', type=int, default=150,
                        help='number of warmup steps')
    args = parser.parse_args()

    # Set up tracking
    accelerator.init_trackers(project_name="sft")

    # SFT model will be adversary
    adversary = LanguageModel(dont_init=True)
    adversary.model = AutoModelForCausalLM.from_pretrained(args.weights)
    adversary.tokenizer = AutoTokenizer.from_pretrained(args.weights)
    
    # Init defender
    defender = LanguageModel(dont_init=True)
    defender.model = AutoModelForCausalLM.from_pretrained(args.defense)
    defender.tokenizer = AutoTokenizer.from_pretrained(args.defense)
    defender.model.eval()
    
    # GPT 2 doesn't have a padding token, so we add it
    adversary.tokenizer.pad_token = adversary.tokenizer.eos_token
    defender.tokenizer.pad_token = defender.tokenizer.eos_token
    adversary.tokenizer.pad_token_id = adversary.tokenizer.eos_token_id
    defender.tokenizer.pad_token_id = defender.tokenizer.eos_token_id
    
    # Optimizer and scheduler
    # Default LR from HF SFTConfig
    optimizer = AdamW(adversary.model.parameters(), lr=5e-05)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (args.warmup_steps + 1)))
   
    # In case accelerator moves things around
    (adversary.model, defender.model, optimizer, scheduler) = accelerator.prepare(adversary.model, defender.model, optimizer, scheduler)

    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)

    # Save the starting state
    accelerator.save_state("sft/")
    
    print("cuda?",torch.cuda.is_available)
    
    # Watch model
    wandb.watch(adversary.model)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=adversary.tokenizer, mlm=False)
    tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, adversary.tokenizer),
    )
    print(tokenized_dataset)

    wandb.log({"test":1})
    wandb.watch(adversary.model)

    training_args = TrainingArguments(
        output_dir="sft_out/",
        learning_rate=5e-5,
        logging_first_step=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=adversary.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    """
    sft_config = SFTConfig(
        max_seq_length=512,
        output_dir="sft_out/",
        logging_steps=10,
        report_to="wandb",
        run_name="baseline"
    )
    trainer = SFTTrainer(
        model="openai-community/gpt2", #adversary.model,
        train_dataset=dataset["train"],
        formatting_func=formatting_prompts_func,
        args=sft_config
    )
    """
    trainer.train()
    accelerator.end_training()
