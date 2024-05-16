from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

checkpoint = "./models/ppo_model_toxbase_checkpoint"

model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint)
model_base = AutoModelForCausalLMWithValueHead.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

while True:
    r = input("> ").strip()
    t = tokenizer(r, return_tensors="pt")

    res = model.generate(**t, max_new_tokens=48, temperature=0.7, top_p=0.9, do_sample=True)
    res_orig = model_base.generate(**t, max_new_tokens=48, temperature=0.7, top_p=0.9, do_sample=True)

    res = tokenizer.batch_decode(res)[0]
    res_orig = tokenizer.batch_decode(res_orig)[0]

    print("==== POLICY ====")
    print(res)
    print("==== BASE ====")
    print(res_orig)

    breakpoint()


