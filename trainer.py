from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from lm import *
from environment import *
import torch

logger = get_logger("ast")

class Trainer:

    def __init__(self,
                 horizon=5,
                 model="openai-community/gpt2",
                 defense="openai-community/gpt2",
                 **kwargs):

        config = PPOConfig(
            model_name=model,
            learning_rate=1.41e-5,
            mini_batch_size=horizon,
            batch_size=horizon,
            **kwargs
        )

        # because the PPO wrapper chops the end off and add
        # a value head, we can't just naively initalize a GPT-2
        # supposedly, though, APIs are the same so we can
        # just use it in our inference wrapper

        self.adversary = LanguageModel(dont_init=True)
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


        self.ppo = PPOTrainer(
            model = adversary_model,
            tokenizer = self.adversary.tokenizer,
            config = config
        )

        # because the accelerator may move models to weird places, we 
        # account for that
        self.adversary.model = self.ppo.model
        self.defender.model = self.ppo.accelerator.prepare(self.defender.model)

        self.horizon = horizon

    def prepare(self, prompts):
        """Make a distributed dataset from stings for training.

        Parameters
        ----------
        prompts : List[List[str]]
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
                return self.__data[x]
            def __len__(self):
                return len(self.__data)

        ds = TrainerDataset(prompts)
        # batch_size = 1 because we will blow each batch
        # up to an entire dialogue
        dl = DataLoader(ds, 1) 

        # huggingface accelerate may ship the dataset
        # off to different processes, etc.
        return self.ppo.accelerator.prepare(dl)

    def epoch(self, dataloader, id=""):
        """Run an epoch of the data.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader you got from self.prepare.
        id : Optional[str]
            ID of this epoch for logging purposes.
        """
        
        
        # this should be a batch of one, so we index
        # to get rid of the outer shell
        for batch in dataloader:
            self.step(batch[0])

        # todo eval
        logger.info(f"Done with epoch {id}".strip())

    def step(self, prompt):
        """Optimize our model by a single step.

        Parameters
        ----------
        prompt : List[str]
            The prompt to optimize!
        """
        
        # run the prompt
        eps, rewards, _ = episode(self.adversary, self.defender, prompt,
                                  horizon=self.horizon, device=self.ppo.accelerator.device)
        rewards = torch.tensor(rewards)

        # the environment has already prepared query and response
        # tensors for us. to edit that behavior, change the environment
        qs = [i.query for i in eps]
        rs = [i.response for i in eps]

        print(eps[-1].ast_utt)

        # get input IDs for queries and responses, padded
        query_ids = self.adversary.tokenizer(qs)["input_ids"]
        response_ids = self.adversary.tokenizer(rs)["input_ids"]

        # Run PPO step
        stats = self.ppo.step([torch.tensor(i) for i in query_ids],
                              [torch.tensor(i) for i in response_ids],
                              list(rewards.unbind(0)))

        # we need to send rewards to cuda because ddp needs them on the
        # same device for logging
        self.ppo.log_stats(stats, {"query": qs, "response": rs}, 
                           rewards.to(self.ppo.accelerator.device))

if __name__ == "__main__":
    accelerator_kwargs = {
        # "cpu": True
    }
    # initialize accelerator once before??
    acc = Accelerator(**accelerator_kwargs)
    trainer = Trainer(accelerator_kwargs=accelerator_kwargs)

    # make some mock data
    prompts = [
        ["WTF are you doing?"],
    ]

    dl = trainer.prepare(prompts)

    # train on a single line
    trainer.epoch(dl)
    breakpoint()
