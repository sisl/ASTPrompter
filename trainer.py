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

logger = get_logger("ast")

class Trainer:

    def __init__(self,
                 args,
                 model="openai-community/gpt2",
                 defense="openai-community/gpt2",
                 **kwargs):

        # cache horizon
        self.horizon = args.horizon

        # initialize early the accelator
        self.accelerator = Accelerator(**kwargs.get(accelerator_kwargs, {}),
                                       log_with="wandb" if args.wandb else None)
        if args.wandb:
            self.accelerator.init_trackers(
                project_name=args.wandb_project_name, 
                config=vars(args),
                init_kwargs=args.wandb_kwargs
            )

        # because the PPO wrapper chops the end off and add
        # a value head, we can't just naively initalize a GPT-2
        # supposedly, though, APIs are the same so we can
        # just use it in our inference wrapper

        self.adversary = LanguageModel(dont_init=True)
        if args.warm_start:
            adversary_model = AutoModelForCausalLM.from_pretrained(args.warm_start)
        else:
            adversary_model = AutoModelForCausalLM.from_pretrained(config.model_name)
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

        # all the mishmash to get
        self.beta = args.beta
        optimizer = AdamW(self.adversary.model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (args.warmup_steps + 1)))

        # because the accelerator may move models to weird places, we 
        # account for that
        (self.adversary.model, self.defender.model,
         self.optimizer, self.scheduler) = self.ppo.accelerator.prepare(self.defender.model, adversary_model,
                                                                        optimizer, scheduler)

        save_name = f"dpo_model_{model.split('/')[-1]}"
        if args.save_name:
            save_name = args.save_name
        self.save_dir = os.path.join(args.save_dir, save_name)
        self.args = args
        self.global_step_counter_ = 0

    def save(self, postfix=""):
        """save the model, optionally with a postfix

        Parameters
        ----------
        postfix : str
            the postfix to save (i.e. if save name was this/here, postfix
            will make it this/here_postfix)
        """

        self.adversary.model.save_pretrained((self.save_dir+("_"+postfix if postfix != "" else "").strip()))
        self.adversary.tokenizer.save_pretrained((self.save_dir+("_"+postfix if postfix != "" else "").strip()))
    
    def prepare(self, steps, batch=1):
        """Make a distributed dataset from stings for training.

        Parameters
        ----------
        steps : List[ASTStep]
            Prompt strings.

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
                return (self.__data[x].query+self.__data[x].response_w,
                        self.__data[x].query+self.__data[x].response_l)
            def __len__(self):
                return len(self.__data)

        ds = TrainerDataset(steps)
        # batch_size = 1 because we will blow each batch
        # up to an entire dialogue
        dl = DataLoader(ds, batch) 

        # huggingface accelerate may ship the dataset
        # off to different processes, etc.
        return self.accelerator.prepare(dl)

    def play(self, prompt):
        return episode_paired(self.adversary, self.defender, prompt, self.horizon)

    def epoch(self, dataloader, log_every=10):
        """Run an epoch of the data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader you got from self.prepare.
        log_every : Optional[int]
            how often to log to wandb
        """
        
        for i, batch in enumerate(tqdm(iter(dataloader), total=len(dataloader))):
            loss, metrics = self.step(batch, log=(i % log_every == 0))
            self.accelerator.backward(loss / self.args.accumulate_steps)

            if (i % self.args.accumulate_steps) == 0:
                gn = clip_grad_norm_(self.adversary.model.parameters(), self.args.max_gradient_norm).cpu().item()
                metrics["training/gradient_norm"] = gn
                self.optimizer.step()
                self.scheduler.step()

            if (i % log_every == 0):
                self.accelerator.log(metrics, step=self.global_step_counter_)

            self.global_step_counter_ += 1

    def finish(self):
        self.accelerator.end_training()

    def __loss(self, policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps):
        # https://github.com/eric-mitchell/direct-preference-optimization/blob/ \
        # f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}


        # this is IPO, Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        losses = (logits - 1/(2 * self.beta)) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loses, chosen_rewards, rejected_rewards

    def step(self, batch, log=False):
        """Optimize our model by a single step.

        Parameters
        ----------
        batch : (List[ASTStep], List[rewards])
            The prompt to optimize!
        log : bool
            whether to log
        """

        combined_wins, combined_loses = batch

        # we need to individualy calculate the logprobs of wins and loses
        # for both our adversarial model + defender model
        with torch.inference_mode():
            defender_logprobs_win = self.defender.logprob_batched(combined_wins, self.accelerator.device)
            defender_logprobs_loss = self.defender.logprob_batched(combined_loses, self.accelerator.device)
        adversary_logprobs_win = self.adversary.logprob_batched(combined_wins, self.accelerator.device) 
        adversary_logprobs_loss = self.adversary.logprob_batched(combined_loses, self.accelerator.device) 

        # yipee
        loses, chosen_rewards, rejected_rewards = self.__loss(adversary_logprobs_win,
                                                              adversary_logprobs_loss,
                                                              defender_logprobs_win,
                                                              defender_logprobs_loss)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics = {
            "rewards/chosen": chosen_rewards.cpu().numpy().tolist(),
            "rewards/rejected": rejected_rewards.cpu().numpy().tolist(),
            "rewards/reward_accuracy": reward_accuracies.cpu().numpy().tolist(),
            "rewards/reward_margin": (chosen_rewards - rejected_rewards).cpu().numpy().tolist(),
            "policy/logprobs_chosen": adversary_logprobs_win.cpu().numpy().tolist(),
            "policy/logprobs_rejected": adversary_logprobs_loss.cpu().numpy().tolist(),
            "ref/logprobs_chosen": defender_logprobs_win.cpu().numpy().tolist(),
            "ref/logprobs_rejected": defender_logprobs_loss.cpu().numpy().tolist(),
            "training/loss": loses.mean().cpu().item(),
        }

        return loses.mean(), metrics

#         combined_wins, combined_loses


#         rewards = torch.tensor(rewards_list).float()

# #         if any(rewards > 2):
#             # breakpoint()

#         # get input IDs for queries and responses, padded
#         query_ids = self.adversary.tokenizer(qs)["input_ids"]
#         response_ids = self.adversary.tokenizer(rs)["input_ids"]

#         # trl isn't happy if we have a batch size that don't match
#         if len(query_ids) != self.batch_size:
#             breakpoint()
#             return
    
#         # if the AST said nothing, don't run anything 
#         if 0 in [len(i) for i in response_ids]:
#             breakpoint()
#             return

#         # Run PPO step
#         # try:
#         # we want a list, one for each batch elemnet for the batch
#         # also, we crop the input in case they are too long to fix the context
#         # we proirtize keeping the end of the input and the beginning of
#         # the output
#         stats = self.ppo.step([torch.tensor(i)[-959:] for i in query_ids],
#                               [torch.tensor(i)[:64] for i in response_ids],
#                               list(rewards.unbind(0)))
#         # except RuntimeError as e:
#             # return

#         # we need to send rewards to cuda because ddp needs them on the
#         # same device for logging
#         if log:
#             self.ppo.log_stats(stats, {"query": qs, "response": rs}, 
#                             rewards.to(self.accelerator.device))
#             table = wandb.Table(columns=["prompt", "ast", "defense", "reward"],
#                                 rows=[[i, j, k, r] 
#                                     for i,j,k,r in zip(p_ut, a_ut, def_ut, rewards_list)])
#             self.accelerator.log({"debug/pairings": table})

