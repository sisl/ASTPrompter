from convokit import Corpus, download, Conversation
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from accelerate import Accelerator
from accelerate.utils import set_seed
from trainer import Trainer

import random
import argparse
import json
import torch

import os
import csv
from datasets import load_dataset
from dotenv import load_dotenv

from accelerate.logging import get_logger
from accelerate.utils.tqdm import tqdm

from datasets import Dataset

import os

os.environ["WANDB_PROJECT"] = "ast"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

logger = get_logger("ast")

# Get token for Toxi-Gen prompts
load_dotenv()
TOKEN = os.environ.get("HF_GIT_TOKEN")

# fix random sede for reproducibility
R = random.Random(24)
# to test if its data dependent
# R = random.Random(54)

# TEACH = False

# if not TEACH:
# load our initial corpus ahead of time
corpus = Corpus(filename=download("reddit-corpus-small"))
id2results = jsonl_to_dict('detox_results.jsonl')
corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
corpus = filter_corpus_formatting(corpus)
convos = list(corpus.conversations.values())

# we only keep the last five utterances (and also discard the front
# because the front is the self-post on reddit)
prompts = [[clean_utterance(j.text, R)
            for j in list(i.iter_utterances())
            if j.text.strip() != "[deleted]"
            and j.text.strip() != ""][1:][-2:]
           for i in convos]
prompts = [[j for j in i if j.strip() != ""]
        for i in prompts]
prompts = [i for i in prompts if len(i) != 0]


with open("prompts.jsonl", 'r') as df:
    lines = df.readlines()
    data = json.loads("["+",".join(lines)+"]")
    # prompts = [i["prompt"]["text"] for i in data if i["challenging"] == True]
    # the random choice is to randomly seed
    # initial capitaliszation. RTP starts with captial letter
    # always which confuses the model
    prompts_rtp = [(R.choice([i["prompt"]["text"][0].lower(),
        i["prompt"]["text"][0]])+i["prompt"]["text"][1:], 
        R.choice([i["continuation"]["text"][0].lower(),
            i["continuation"]["text"][0]])+i["continuation"]["text"][1:])
        for i in data if i["continuation"]["toxicity"]
        and i["continuation"]["toxicity"] > 0.3]

prompts_tox_comments = key_list = list(map(lambda row: row[0], csv.reader(open('toxic_comments.csv'))))
    

# fire this puppy off 
if __name__ == "__main__":
    # set random seed for reproducability
    set_seed(24)

    # establish the arguments of this system
    parser = argparse.ArgumentParser(description='AST Trainer')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='each batch will be batch_size*accumulate_steps')
    parser.add_argument('--horizon', type=int, default=3,
                        help='how many turns to self-play')
    parser.add_argument('--tox_mix', type=float, default=0.5,
                        help='for how many EPISODES do we mix in a single toxicity prompt?')
    parser.add_argument('--experience_size', type=int, default=512,
                        help='how many experience samples to collect per epoch?')
    parser.add_argument('--lr', type=float, default=5e-7,
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='IPO/DPO beta')
    parser.add_argument('--accumulate_steps', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--max_gradient_norm', type=float, default=10,
                        help='maximum gradient norm to clip to')
    parser.add_argument('--warmup_steps', type=int, default=150,
                        help='number of warmup steps')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='prefix of the model save dir, default "models"')
    parser.add_argument('--save_name', type=str, default=None,
                        help='the folder place to save our model')
    parser.add_argument('--warm_start', type=str, default=None,
                        help='start your policy here')
    parser.add_argument('--defense', type=str, default=None,
                        help='start your defense here')
    parser.add_argument('--wandb', action="store_true", default=False,
                        help='use wandb?')
    args = parser.parse_args()

    # if we are CPU, we have to do it here BEFORE argparse
    accelerator_kwargs = {
        # "cpu": True
    }

    # initialize accelerator once before??
    trainer = Trainer(args,
                      accelerator_kwargs=accelerator_kwargs,
                      wandb_project_name="ast",
                      wandb_kwargs={
                          "wandb": {
                              "entity": "jemoka", 
                              # comment the line below out to log
                              "mode": None if args.wandb else "disabled"
                          }
                      },
                      model_load_params={
                          # "load_in_8bit": True,
                          # "attn_implementation": "flash_attention_2",
                          # "torch_dtype": torch.float16
                          # "gradient_checkpointing": True
                      })
                      # ref="openai-community/gpt2")

    ##########

    # good vibes time
    for epoch in range(args.epochs):
        # shuffle the data
        R.shuffle(prompts)
        # experience the experience
        with trainer.accelerator.main_process_first():
            logger.info("loading training data...")

            # IF we are currently teaching, collect teaching trajectories
            steps = []

            # we will keep rolling out until we get experience size
            with tqdm(total=args.experience_size) as bar:
                while len(steps) < args.experience_size:
                    # check if we want to insert a teaching statement
                    if R.random() < args.tox_mix:
                        steps.append(trainer.teach(R.choice(prompts_rtp)[0]))
                        bar.update(1)
                    else:
                        try:
                            step = trainer.play(R.choice(prompts))
                            bar.update(len(step))
                            steps += step
                        except RuntimeError:
                            continue
        # 
        # breakpoint()

        # on *EACH THREAD*, prepare our dataset
        dataset = trainer.prepare(steps, batch=args.batch_size)
        
        # replay the experience
        trainer.epoch(dataset, log_every=10)

        # have a good time

        # epoch_rewards = []

        # # run the validation prompts and get mean reward
        # for i in val_dl:
            # # recall each has a batch size of 1
            # _, rew, convo = trainer.play(i[0])
            # # tally the reward for averaging in the end
            # epoch_rewards += rew

        # # log!
        # epoch_reward = sum(epoch_rewards)/len(epoch_rewards)
        # trainer.accelerator.log({"validation_reward": epoch_reward})
        # print(f"reward: {epoch_reward}")

        # # print("\n".join(convo))

        # # if we are at best epoch, save best weights, othrewise,
        # # we still checkpoint every epoch
        # if best_reward < epoch_reward:
            # trainer.save("best")
            # best_reward = epoch_reward

        trainer.save("checkpoint")

    trainer.finish()
