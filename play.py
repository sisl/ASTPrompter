from transformers import AutoTokenizer, AutoModelForCausalLM
from lm import LanguageModel
from environment import episode

checkpoint = "./models/dpo_model_gpt2_checkpoint"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
model_base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

adversary = LanguageModel(dont_init=True)
adversary.model = model
adversary.tokenizer = tokenizer

defender = LanguageModel(dont_init=True)
defender.model = model_base
defender.tokenizer = tokenizer

while True:
    prompt = []
    r = None
    while r != 'q':
        r = input("> ").strip()
        if r != "q":
            prompt.append(r)
    convo_policy = episode(adversary, defender, prompt, horizon=3)
    convo_base = episode(defender, defender, prompt, horizon=3)

    print("==== POLICY ====")
    print(" ".join(convo_policy))
    print("==== BASE ====")
    print(" ".join(convo_base))

    breakpoint()


