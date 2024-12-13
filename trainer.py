from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import torch.nn.functional as F
from lm import *
from peft import LoraConfig
from environment import *
import torch
from torch.nn.utils import clip_grad_norm_
import os
import wandb
import json
from accelerate.state import PartialState

logger = get_logger("ast")

class Trainer:

    def __init__(self, args, **kwargs):

        # cache horizon
        self.horizon = args.horizon

        # initialize early the accelator
        self.accelerator = Accelerator(**kwargs.get("accelerator_kwargs", {}),
                                       log_with="wandb" if args.wandb else None)
        if args.wandb:
            self.accelerator.init_trackers(
                project_name="ast", 
                config=vars(args),
                init_kwargs=kwargs.get("wandb_kwargs", {})
            )

        # because the PPO wrapper chops the end off and add
        # a value head, we can't just naively initalize a GPT-2
        # supposedly, though, APIs are the same so we can
        # just use it in our inference wrapper

        self.adversary = LanguageModel(dont_init=True)
        self.adversary.model = AutoModelForCausalLM.from_pretrained(args.adversary, **kwargs.get("model_load_params", {}))
        self.adversary.tokenizer = AutoTokenizer.from_pretrained(args.adversary)

        if args.defense == args.baseline:
            # freeze a copy of the model and initialize both defense and train with it to save space
            frozen_model = AutoModelForCausalLM.from_pretrained(args.adversary, **kwargs.get("model_load_params", {}))
            frozen_tokenizer = AutoTokenizer.from_pretrained(args.adversary)

            self.defender = LanguageModel(dont_init=True)
            self.defender.model = frozen_model
            self.defender.tokenizer = frozen_tokenizer

            self.baseline = LanguageModel(dont_init=True)
            self.baseline.model = frozen_model
            self.baseline.tokenizer = frozen_tokenizer

            self.defender.model.eval()
            self.baseline.model.eval()
        else:
            # our defender can be initialized normally 
            # and immeditaley frozen
            self.defender = LanguageModel(args.defense,
                                        model_load_params=kwargs.get("model_load_params", {}))
            self.defender.model.eval()
            self.baseline = LanguageModel(args.baseline,
                                        model_load_params=kwargs.get("model_load_params", {}))
            self.baseline.model.eval()

        # GPT 2 doesn't have a padding token, so we add it
        if "gpt2" in args.adversary:
            self.adversary.tokenizer.pad_token = self.adversary.tokenizer.eos_token
            self.adversary.tokenizer.pad_token_id = self.adversary.tokenizer.eos_token_id

        if "gpt2" in args.defense:
            self.defender.tokenizer.pad_token = self.defender.tokenizer.eos_token
            self.defender.tokenizer.pad_token_id = self.defender.tokenizer.eos_tooken_id

        if "gpt2" in args.baseline:
            self.baseline.tokenizer.pad_token = self.baseline.tokenizer.eos_token
            self.baseline.tokenizer.pad_token_id = self.baseline.tokenizer.eos_token_id

        # all the mishmash to get
        self.beta = args.beta
        optimizer = AdamW(self.adversary.model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (args.warmup_steps + 1)))

        # because the accelerator may move models to weird places, we 
        # account for that
        (self.adversary.model, self.defender.model,
         self.baseline.model, self.optimizer, self.scheduler) = self.accelerator.prepare(self.adversary.model, self.defender.model,
                                                                                         self.baseline.model, optimizer, scheduler)
        if args.wandb:
            wandb.watch(self.adversary.model)

        save_name = f"model_{args.adversary.split('/')[-1]}_{args.defense.split('/')[-1]}"
        if args.save_name:
            save_name = args.save_name
        self.save_dir = os.path.join(args.save_dir, save_name)
        self.args = args

        self.global_step_counter_ = 0

    @classmethod
    def warm_start(cls, args, state_path, **kwargs):
        """Warm start the trainer using the arguments and state from another path.

        Parameters
        ----------
        args : Namespace
            Default arguments, if none are provided.
        state_path : str
            The path to restore state from.

        Returns
        -------
        Trainer, Any
            The restored trainer, and any metadata that originally
            provided to `self.save`'s `entire_state` argument..
        """

        state = PartialState()
        
        try:
            # load the arguments state first
            with open(os.path.join(state_path, "meta.json"), 'r') as df:
                state = json.load(df)
        except FileNotFoundError as e:
            logger.warning("Saved checkppoint not found, starting from scratch!")
            trainer = cls(args, **kwargs)
            return trainer, {}

        if args.__dict__ != state["arguments"]:
            logger.warning("Loaded checkpoint has different args than args provided, defaulting to LOADED args!")
        args.__dict__.update(state["arguments"])

        trainer = cls(args, **kwargs)
        trainer.global_step_counter_ = state["steps"]
        trainer.accelerator.load_state(state_path)

        return trainer, dict(state["train_state"])
   
    def save(self, postfix="", entire_state=None):
        """save the model, optionally with a postfix

        Parameters
        ----------
        postfix : str
            the postfix to save (i.e. if save name was this/here, postfix
            will make it this/here_postfix)
        entire_state : Optional[Dict]
            save training state, used for warm starts 
        """

        savedir = (self.save_dir+("_"+postfix if postfix != "" else "").strip())

        if not entire_state:
            self.adversary.model.save_pretrained(savedir)
            self.adversary.tokenizer.save_pretrained(savedir)
        else:
            arguments = vars(self.args)
            self.accelerator.save_state(savedir, safe_serialization=False)
            with open(os.path.join(savedir, "meta.json"), 'w') as df:
                json.dump({
                    "arguments": arguments,
                    "train_state": entire_state,
                    "steps": self.global_step_counter_,
                }, df, indent=4)

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
            def __init__(self, data):
                super().__init__()
                self.__data = data
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
        return episode_paired(self.adversary, self.defender, [i+" " for i in prompt], 
                self.horizon, difference_threshold=self.args.threshold, 
                              reward_options={"ast_ppl_weight": self.args.ast_ppl_weight})

    def teach(self, prompt):
        return teach_paired(self.adversary, self.defender, prompt, 
                            reward_options={"ast_ppl_weight": self.args.ast_ppl_weight})

    def episode(self, prompt):
        return episode(self.adversary, self.defender, [i+" " for i in prompt], self.horizon,
                       reward_options={"ast_ppl_weight": self.args.ast_ppl_weight})

    def rollout(self, prompt, **kwargs):
        current_prompt = prompt
        def_ut = self.adversary.rollout(current_prompt, max_new_tokens=24, repetition_penalty=1.1,
                                        temperature=0.7, skip_special_tokens=True, **kwargs)
        def_ut = def_ut.replace(current_prompt, "")

        return def_ut

    def epoch(self, dataloader, log_every=10):
        """Run an epoch of the data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader you got from self.prepare.
        log_every : Optional[int]
            how often to log to wandb
        """
        
        for i, batch in enumerate(iter(dataloader)):
            loss, metrics = self.step(batch, log=(i % log_every == 0))
            loss = loss / self.args.accumulate_steps
            if not torch.isnan(loss):
                self.accelerator.backward(loss)

            if (i % self.args.accumulate_steps) == 0:
                gn = clip_grad_norm_(self.adversary.model.parameters(), self.args.max_gradient_norm).cpu().item()
                metrics["training/gradient_norm"] = gn
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (i % log_every == 0):
                metrics["training/lr"] = self.optimizer.param_groups[0]["lr"]
                self.accelerator.log(metrics, step=self.global_step_counter_)
                logger.info(f"MARGIN {round(metrics['rewards/reward_margin'],5)} and LOSS {round(metrics['training/loss'],2)} in STEP {self.global_step_counter_}/{self.args.total_steps}")

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
        if self.args.dpo:
            losses = -F.logsigmoid(self.beta * logits) * (1 - self.args.label_smooth) - F.logsigmoid(-self.beta * logits) * self.args.label_smooth
        else:
            losses = (logits - 1/(2 * self.beta)) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

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
            defender_logprobs_win = self.baseline.logprob_batched(combined_wins, self.accelerator.device)
            defender_logprobs_loss = self.baseline.logprob_batched(combined_loses, self.accelerator.device)
        adversary_logprobs_win = self.adversary.logprob_batched(combined_wins, self.accelerator.device) 
        adversary_logprobs_loss = self.adversary.logprob_batched(combined_loses, self.accelerator.device) 

        # yipee
        loses, chosen_rewards, rejected_rewards = self.__loss(adversary_logprobs_win,
                                                              adversary_logprobs_loss,
                                                              defender_logprobs_win,
                                                              defender_logprobs_loss)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if torch.isnan(loses.mean()):
            breakpoint()

        metrics = {
            "rewards/chosen": chosen_rewards.mean().cpu().item(),
            "rewards/rejected": rejected_rewards.mean().cpu().item(),
            "rewards/reward_accuracy": reward_accuracies.mean().cpu().item(),
            "rewards/reward_margin": (chosen_rewards - rejected_rewards).mean().cpu().item(),
            "policy/logprobs_chosen": adversary_logprobs_win.mean().detach().cpu().item(),
            "policy/logprobs_rejected": adversary_logprobs_loss.mean().detach().cpu().item(),
            "ref/logprobs_chosen": defender_logprobs_win.mean().detach().cpu().item(),
            "ref/logprobs_rejected": defender_logprobs_loss.mean().detach().cpu().item(),
            "training/loss": loses.mean().detach().cpu().item(),
            "debug/text": wandb.Table(data=list(zip(combined_wins, combined_loses)), 
                columns=["chosen", "rejected"])
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

