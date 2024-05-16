from convokit import Corpus, download, Conversation
from toxicity.detoxify_reddit import filter_corpus_toxicity, jsonl_to_dict
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance

from accelerate import Accelerator
from trainer import Trainer

import random
import argparse
import json

from accelerate.logging import get_logger
from accelerate.utils.tqdm import tqdm

logger = get_logger("ast")

# fix random sede for reproducibility
R = random.Random(24)

TEACH = False

if not TEACH:
    # load our initial corpus ahead of time
    corpus = Corpus(filename=download("reddit-corpus-small"))
    id2results = jsonl_to_dict('detox_results.jsonl')
    corpus = filter_corpus_toxicity(corpus, id2results, {"toxicity": 0.5})
    corpus = filter_corpus_formatting(corpus)
    convos = list(corpus.conversations.values())

    # we only keep the last five utterances (and also discard the front
    # because the front is the self-post on reddit)
    prompts = [[clean_utterance(j.text) 
                for j in list(i.iter_utterances()) if j.text.strip() != "[deleted]"][1:][-5:]
                for i in convos]
else:
    with open("prompts.jsonl", 'r') as df:
        lines = df.readlines()
        data = json.loads("["+",".join(lines)+"]")
        # prompts = [i["prompt"]["text"] for i in data if i["challenging"] == True]
        prompts = [(i["prompt"]["text"], 
                    i["continuation"]["text"])
                   for i in data if i["continuation"]["toxicity"]
                   and i["continuation"]["toxicity"] > 0.3]

# fire this puppy off 
if __name__ == "__main__":
    # establish the arguments of this system
    parser = argparse.ArgumentParser(description='AST Trainer')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--horizon', type=int, default=4,
                        help='horizon of each episode')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='horizon of each episode')
    parser.add_argument('--experience_size', type=int, default=1024,
                        help='how many experience samples to collect per epoch?')
    parser.add_argument('--lr', type=float, default=1.41e-6,
                        help='learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='prefix of the model save dir, default "models"')
    parser.add_argument('--save_name', type=str, default=None,
                        help='the folder place to save our model')
    args = parser.parse_args()

    # if we are CPU, we have to do it here BEFORE argparse
    accelerator_kwargs = {
        # "cpu": True
    }

    # initialize accelerator once before??
    trainer = Trainer(args,
                      accelerator_kwargs=accelerator_kwargs,
                      log_with="wandb",
                      tracker_project_name="ast",
                      tracker_kwargs={
                          "wandb": {
                              "entity": "jemoka", 
                              # comment the line below out to log
                              "mode": "disabled"
                          }
                      })

    ##########

    best_reward = float("-inf")
    
    # good vibes time
    for epoch in range(args.epochs):
        # shuffle the data
        R.shuffle(prompts)
        # experience the experience
        with trainer.accelerator.main_process_first():
            logger.info("loading training data...")

            # IF we are currently teaching, collect teaching trajectories
            steps, rewards = [], []

            if TEACH:
                # recall prompts are shuffled
                for prompt in tqdm(prompts[:args.experience_size]):
                    step, reward = trainer.teach(*prompt)
                    steps.append(step)
                    rewards.append(reward)
            else:
                # we will keep rolling out until we get experience size
                with tqdm(total=args.experience_size) as bar:
                    while len(steps) < args.experience_size:
                        step, reward, _ = trainer.play(R.choice(prompts))
                        steps += step
                        rewards += reward
                        bar.update(len(step))

        # on *EACH THREAD*, prepare our dataset
        dataset = trainer.prepare(steps, rewards, batch=args.batch_size)
        
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
