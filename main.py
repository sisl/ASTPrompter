from convokit import Corpus, download, Conversation
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance, corpus_len, corpus_to_prompts
from toxicity.split_data import filter_corpus_by_file

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

import logging

os.environ["WANDB_PROJECT"] = "ast"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

logger = get_logger("ast")

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.ERROR)
logger.setLevel(logging.DEBUG)

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
train_corp = filter_corpus_by_file(Corpus(filename=download("reddit-corpus-small")), "data/train.txt")
dev_corp = filter_corpus_by_file(Corpus(filename=download("reddit-corpus-small")), "data/dev.txt")

# corpus -> prompts
train_prompts = corpus_to_prompts(train_corp)
dev_prompts = corpus_to_prompts(dev_corp)
#dev_prompts = corpus_to_prompts(dev_corp)
#test_prompts = corpus_to_prompts(test_corp)

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
                   and i["continuation"]["toxicity"] > 0.5]

# fire this puppy off 
if __name__ == "__main__":
    # set random seed for reproducability
    set_seed(24)

    # establish the arguments of this system
    parser = argparse.ArgumentParser(description='AST Trainer')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='each batch will be batch_size*accumulate_steps')
    parser.add_argument('--horizon', type=int, default=3,
                        help='how many turns to self-play')
    parser.add_argument('--tox_mix', type=float, default=0.5,
                        help='for how many EPISODES do we mix in a single toxicity prompt?')
    parser.add_argument('--threshold', type=float, default=0,
                        help='how different does a pair have to be to count?')
    parser.add_argument('--experience_size', type=int, default=512,
                        help='how many experience samples to collect per epoch?')
    parser.add_argument('--lr', type=float, default=5e-7,
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='IPO/DPO beta')
    parser.add_argument('--accumulate_steps', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--max_gradient_norm', type=float, default=10,
                        help='maximum gradient norm to clip to')
    parser.add_argument('--warmup_steps', type=int, default=150,
                        help='number of warmup steps')
    parser.add_argument('--ast_ppl_weight', type=float, default=0.1,
                        help='the weight on the perplexity term, higher means more likely')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='evaluate model every this many epochs')
    parser.add_argument('--total_steps', type=int, default=10000,
                        help='total steps to train')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='prefix of the model save dir, default "models"')
    parser.add_argument('--save_name', type=str, required=True,
                        help='the folder place to save our model')
    parser.add_argument('--adversary', type=str, default='openai-community/gpt2',
                        help='start your policy here')
    parser.add_argument('--baseline', type=str, default='openai-community/gpt2',
                        help='use this as your baseline model for ipo')
    parser.add_argument('--defense', type=str, default='openai-community/gpt2',
                        help='defense model')
    parser.add_argument('--warm_start', type=str, default=None,
                        help='start your model warm from this checkpoint')
    parser.add_argument('--wandb', action="store_true", default=False,
                        help='use wandb?')
    parser.add_argument('--dpo', action="store_true", default=False,
                        help='use dpo?')
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='cdpo label smooth, not used in ipo')
    args = parser.parse_args()

    # if we are CPU, we have to do it here BEFORE argparse
    accelerator_kwargs = {
        # "cpu": True
    }

    # initialize accelerator once before??
    if not args.warm_start:
        trainer = Trainer(args,
                        accelerator_kwargs=accelerator_kwargs,
                        wandb_project_name="ast",
                        wandb_kwargs={
                            "wandb": {
                                "entity": "jemoka", 
                                "mode": None if args.wandb else "disabled",
                                "name": args.save_name
                            }
                        },
                        model_load_params={
                            # "load_in_8bit": True,
                            # "attn_implementation": "flash_attention_2",
                            # "torch_dtype": torch.float16
                            # "gradient_checkpointing": True
                        })
                      # ref="openai-community/gpt2")
        meta = {}
    else:
        trainer, meta = Trainer.warm_start(args,
                                           args.warm_start,
                                           accelerator_kwargs=accelerator_kwargs,
                                           wandb_project_name="ast",
                                           wandb_kwargs={
                                               "wandb": {
                                                   "entity": "jemoka", 
                                                   "mode": None if args.wandb else "disabled"
                                               }
                                           },
                                           model_load_params={
                                               # "load_in_8bit": True,
                                               # "attn_implementation": "flash_attention_2",
                                               # "torch_dtype": torch.float16
                                               # "gradient_checkpointing": True
                                    })

    ##########

    # good vibes time
    epoch = meta.get("epoch", 0)
    best_score = meta.get("best", float("-inf"))
    while epoch < args.epochs:
        logger.info(f"EPOCH {epoch} starting...")
        trainer.save("checkpoint", {"epoch": epoch, "best": best_score})

        if epoch % args.eval_every == 0 and epoch != 0:
            logger.info(f"EVALUATING...")
            rewards = []
            for indx, i in enumerate(dev_prompts):
                if indx % 30 == 0:
                    logger.debug(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
                rewards += [j.reward_w for j in trainer.episode(i)]
            logger.debug(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
            dev_score = sum(rewards)/len(rewards)

            if dev_score > best_score:
                logger.info(f"NEW BEST! {round(dev_score, 3)}")
                trainer.accelerator.log({"training/dev_score": dev_score},
                                        step=trainer.global_step_counter_)
                trainer.save("best")

        # shuffle the data
        R.shuffle(train_prompts)
        # experience the experience
        with trainer.accelerator.main_process_first():
            # IF we are currently teaching, collect teaching trajectories
            steps = []

            # we will keep rolling out until we get experience size
            # with tqdm(total=args.experience_size) as bar:
            # `last` is for logging purposes to log every 10 setps or so
            last = 0
            while len(steps) < args.experience_size:
                if last % 50 == 0:
                    logger.debug(f"COLLECTED {len(steps)} < {args.experience_size} steps...")
                last += 1

                # check if we want to insert a teaching statement
                if R.random() < args.tox_mix:
                    steps.append(trainer.teach("".join(R.choice(prompts_rtp))))
                    # bar.update(1)
                else:
                    try:
                        step = trainer.play(R.choice(train_prompts))
                        # bar.update(len(step))
                        steps += step
                    except RuntimeError:
                        continue
            logger.debug(f"COLLECTED {len(steps)} >= {args.experience_size} steps...")

        logger.info(f"{len(steps)} STEPS will be ran in epoch {epoch}...")

        # prepare our sub-dataset of this batch of experience
        dataset = trainer.prepare(steps, batch=args.batch_size)

        logger.info(f"REPLAYING epoch {epoch}...")
        
        # replay the experience and have a good time
        trainer.epoch(dataset, log_every=10)

        if trainer.global_step_counter_ > args.total_steps:
            logger.info(f"FINISHED TRAINING at {trainer.global_step_counter_} steps, breaking...")
            break

        epoch += 1

    if trainer.global_step_counter_ > args.total_steps:
        logger.info(f"TRAINING STOPPED at {epoch} epochs. Bye!")

    logger.info(f"EVALUATING...")
    rewards = []
    for indx, i in enumerate(dev_prompts):
        if indx % 30 == 0:
            logger.debug(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
        rewards += [j.reward_w for j in trainer.episode(i)]
    logger.debug(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
    dev_score = sum(rewards)/len(rewards)

    if dev_score > best_score:
        logger.info(f"NEW BEST! {round(dev_score, 3)}")
        trainer.accelerator.log({"training/dev_score": dev_score},
                                step=trainer.global_step_counter_)
        trainer.save("best")



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


    trainer.finish()
