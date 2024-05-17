from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from lm import *
from peft import LoraConfig
from environment import *
import torch
import os
import wandb

logger = get_logger("ast")

class Trainer:

    def __init__(self,
                 args,
                 model="openai-community/gpt2",
                 defense="openai-community/gpt2",
                 **kwargs):

        horizon = args.horizon

        # this is to calculate the number of total steps
        # which the kl horizon uses to penalize
        kl_horizon = args.epochs * (args.experience_size // args.batch_size)

        config = PPOConfig(
            model_name=model,
            learning_rate=args.lr,
            mini_batch_size=args.batch_size,
            batch_size=args.batch_size,
            kl_penalty="full",
            init_kl_coef=args.init_kl,
            horizon=kl_horizon,
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=args.reward_clip,
            max_grad_norm=args.gradient_clip,
            ratio_threshold=args.ratio_threshold,
            vf_coef=args.vf_scale,
            # whiten_rewards=True,
            # for our problem setting, this seems good?
            # we do want our distribution to deviate quite a bit, but
            # not super much
            # target=4,
            **kwargs
        )

        self.batch_size = args.batch_size

        # because the PPO wrapper chops the end off and add
        # a value head, we can't just naively initalize a GPT-2
        # supposedly, though, APIs are the same so we can
        # just use it in our inference wrapper

        self.adversary = LanguageModel(dont_init=True)
        if args.warm_start:
            adversary_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.warm_start)
        else:
            adversary_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        self.adversary.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # our defender can be initialized normally 
        # and immeditaley frozen
        self.defender = LanguageModel(defense)
        self.defender.model.eval()

        # GPT 2 doesn't have a padding token, so we add it
        self.adversary.tokenizer.pad_token = self.adversary.tokenizer.eos_token
        self.defender.tokenizer.pad_token = self.defender.tokenizer.eos_token
        self.adversary.tokenizer.pad_token_id = self.adversary.tokenizer.eos_token_id
        self.defender.tokenizer.pad_token_id = self.defender.tokenizer.eos_token_id

        self.optimizer = AdamW( 
            filter(lambda p: p.requires_grad, adversary_model.parameters()),
            lr=args.lr,
            # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            eps=1e-5
        )
        self.scheduler = ExponentialLR(self.optimizer, args.decay_factor)

        self.ppo = PPOTrainer(
            model = adversary_model,
            tokenizer = self.adversary.tokenizer,
            config = config,
            optimizer = self.optimizer,
            lr_scheduler = self.scheduler
        )

        # because the accelerator may move models to weird places, we 
        # account for that
        self.adversary.model = self.ppo.model
        self.defender.model = self.ppo.accelerator.prepare(self.defender.model)

        self.horizon = horizon

        save_name = f"ppo_model_{model.split('/')[-1]}"
        if args.save_name:
            save_name = args.save_name
        self.save_dir = os.path.join(args.save_dir, save_name)

    @property
    def accelerator(self):
        return self.ppo.accelerator

    def save(self, postfix=""):
        """save the model, optionally with a postfix

        Parameters
        ----------
        postfix : str
            the postfix to save (i.e. if save name was this/here, postfix
            will make it this/here_postfix)
        """

        self.ppo.save_pretrained((self.save_dir+("_"+postfix
                                                 if postfix != ""
                                                 else "").strip()))
    
    def prepare(self, steps, rewards, batch=1):
        """Make a distributed dataset from stings for training.

        Parameters
        ----------
        steps : List[ASTStep]
            Prompt strings.
        rewards : List[float]
            The rewards which the steps got.

        Returns
        -------
        torch.utils.data.DataLoader
            The dataloader to pass to self.epoch.
        """

        class TrainerDataset(Dataset):
            def __init__(self, data, reward):
                super().__init__()
                self.__data = data
                self.__reward = reward
                assert len(self.__data) == len(self.__reward), "lengths fo reward and data are not the same lengths!"
            def __getitem__(self, x):
                return (self.__data[x].query, self.__data[x].response, self.__reward[x], 
                        self.__data[x].prompt_utt, self.__data[x].ast_utt, self.__data[x].def_utt)
            def __len__(self):
                return len(self.__data)

        ds = TrainerDataset(steps, rewards)
        # batch_size = 1 because we will blow each batch
        # up to an entire dialogue
        dl = DataLoader(ds, batch) 

        # huggingface accelerate may ship the dataset
        # off to different processes, etc.
        return self.accelerator.prepare(dl)

    def epoch(self, dataloader, log_every=10):
        """Run an epoch of the data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader you got from self.prepare.
        log_every : Optional[int]
            how often to log to wandb
        """
        
        # this should be a batch of one, so we index
        # to get rid of the outer shell
        for i, batch in enumerate(tqdm(iter(dataloader), total=len(dataloader))):
            self.step(batch, log=(i % log_every == 0))

    def play(self, prompt):
        """self play to run the prompt

        Parameters
        ----------
        prompt : List[str]
            the prompt to evaluate on

        Returns
        -------
        List[ASTStep], List[float], List[str]
            Steps, rewards per step, conversation.
        """
        
        # run the prompt
        eps, rewards, convo = episode(self.adversary, self.defender, prompt,
                                      horizon=self.horizon, device=self.accelerator.device)

        return eps, rewards, convo

    def teach(self, prompt, response):
        """self play to run the prompt

        Parameters
        ----------
        prompt : str
            the prompt to elicit
        response : str
            what the the AST should respond

        Returns
        -------
        ASTStep, float
            Step, reward.
        """
        
        return teach(self.adversary, self.defender,
                     prompt, response, device=self.accelerator.device)

    def step(self, batch, log=False):
        """Optimize our model by a single step.

        Parameters
        ----------
        batch : (List[ASTStep], List[rewards])
            The prompt to optimize!
        log : bool
            whether to log
        """

        qs, rs, rewards_list, p_ut, a_ut, def_ut = batch
        rewards = torch.tensor(rewards_list).float()

        # get input IDs for queries and responses, padded
        query_ids = self.adversary.tokenizer(qs)["input_ids"]
        response_ids = self.adversary.tokenizer(rs)["input_ids"]

        # trl isn't happy if we have a batch size that don't match
        if len(query_ids) != self.batch_size:
            return
    
        # if the AST said nothing, don't run anything 
        if 0 in [len(i) for i in response_ids]:
            return

        # Run PPO step
        # try:
        # we want a list, one for each batch elemnet for the batch
        # also, we crop the input in case they are too long to fix the context
        # we proirtize keeping the end of the input and the beginning of
        # the output
        stats = self.ppo.step([torch.tensor(i)[-959:] for i in query_ids],
                              [torch.tensor(i)[:64] for i in response_ids],
                              list(rewards.unbind(0)))
        # except RuntimeError as e:
            # return

        # we need to send rewards to cuda because ddp needs them on the
        # same device for logging
        if log:
            self.ppo.log_stats(stats, {"query": qs, "response": rs}, 
                            rewards.to(self.accelerator.device))
            table = wandb.Table(columns=["prompt", "ast", "defense", "reward"],
                                rows=[[i, j, k, r] 
                                    for i,j,k,r in zip(p_ut, a_ut, def_ut, rewards_list)])
            self.accelerator.log({"debug/pairings": table})

