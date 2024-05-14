from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from accelerate import Accelerator
from transformers import AutoTokenizer
from lm import *
from environment import *
import torch

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
        self.adversary.model = adversary_model
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
            model = self.adversary.model,
            tokenizer = self.adversary.tokenizer,
            config = config
        )

        self.horizon = horizon

    def step(self, prompt):
        """Optimize our model by a single step.

        Parameters
        ----------
        prompt : List[str]
            The prompt to optimize!
        """
        
        # run the prompt
        eps, rewards, _ = episode(self.adversary, self.defender, prompt,
                                  horizon=self.horizon)
        rewards = torch.tensor(rewards)

        # the environment has already prepared query and response
        # tensors for us. to edit that behavior, change the environment
        qs = [i.query for i in eps]
        rs = [i.response for i in eps]

        # get input IDs for queries and responses, padded
        query_ids = self.adversary.tokenizer(qs)["input_ids"]
        response_ids = self.adversary.tokenizer(rs)["input_ids"]

        # Run PPO step
        stats = self.ppo.step([torch.tensor(i) for i in query_ids],
                              [torch.tensor(i) for i in response_ids],
                              list(rewards.unbind(0)))
        self.ppo.log_stats(stats, {"query": qs, "response": rs}, rewards)

if __name__ == "__main__":
    accelerator_kwargs = {
        # "cpu": True
    }
    # initialize accelerator once before??
    acc = Accelerator(**accelerator_kwargs)
    trainer = Trainer(accelerator_kwargs=accelerator_kwargs)

    # train on a single line
    trainer.step(["WTF are you doing?"])
    trainer.step(["Who are you?", "You are such a numpty!"])
    breakpoint()
