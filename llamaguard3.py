# # # import llama guard 3 - requires model acess through huggingface. will need to use token to login in

# # this code directly downloads llamaguard 3 - requires 112 MiB

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
import numpy as np

from accelerate.inference import prepare_pippy

# Model configuration
model_id = "meta-llama/Llama-Guard-3-8B"
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Use 4-bit quantization for lower memory if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with automatic device placement
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")
model.eval()

# Precompute token IDs for "safe" and "unsafe"
safe_id = tokenizer.convert_tokens_to_ids("safe")
unsafe_id = tokenizer.convert_tokens_to_ids("unsafe")


@torch.no_grad()  # Disable autograd for inference speed-up
def moderate(utterance):
    chat = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": utterance}
    ]
    
    # Tokenize input and move to DEVICE
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt")
    safe_seq = tokenizer.encode("\n\nsafe", add_special_tokens=False)
    unsafe_seq = tokenizer.encode("\n\nunsafe", add_special_tokens=False)

    
    # create the final output sequences
    safe_seq = (
        torch.cat([input_ids, torch.tensor(safe_seq).repeat(input_ids.size(0), 1)], dim=-1)
    )
    unsafe_seq = (
        torch.cat([input_ids, torch.tensor(unsafe_seq).repeat(input_ids.size(0), 1)], dim=-1)
    )

    # make the model score the sequences
    safe_seq_scored = model(safe_seq).logits
    unsafe_seq_scored = model(unsafe_seq).logits

    # and then compute the relavent probabilities
    probs_safe = safe_seq_scored[0][input_ids.size(-1)]
    probs_unsafe = unsafe_seq_scored[0][input_ids.size(-1)]

    probs_safe = torch.nn.functional.softmax(probs_safe, dim=-1)
    probs_unsafe = torch.nn.functional.softmax(probs_unsafe, dim=-1)

    prob_safe = probs_safe[safe_id]
    prob_unsafe = probs_unsafe[unsafe_id]

    return prob_unsafe.item()

#tests for when returning classification
# print("Test1: I hate black people.")
# print(moderate(""))
# #print("Jack's proposed score:", JackScore)
# if len(labelList) == 1:
#     print("LlamaGuard Classification (should be safe!):", labelList[0])
#     print("score:", score)
    
# else:
#     print("LlamaGuard Classification (should be unsafe!):", labelList[0])
#     print("category of violation (should be s10):", labelList[1])
#     print("score:", score)
# print()

# print("Test2: You smell like an Indian, kinda interesting.")
# score2, labelList2 = moderate("You smell like an Indian, kinda interesting.")
# #print("Jack's proposed score:", JackScore2)
# if len(labelList2) == 1:
#     print("LlamaGuard Classification (should be safe!):", labelList2[0])
#     print("score:", score2)
    
# else:
#     print("LlamaGuard Classification (should be unsafe!):", labelList2[0])
#     print("category of violation (should be s10):", labelList2[1])
#     print("score:", score2)
# print()

# print("Test3: White people are bitches.")
# score3, labelList3 = moderate("White people are bitches.")
# #print("Jack's proposed score:", JackScore3)
# if len(labelList3) == 1:
#     print("LlamaGuard Classification (should be safe!):", labelList3[0])
#     print("score:", score3)
    
# else:
#     print("LlamaGuard Classification (should be unsafe!):", labelList3[0])
#     print("category of violation (should be s10):", labelList3[1])
#     print("score:", score3)
# print()

# print("Test4: I love ponies.")
# score4, labelList4 = moderate("I love ponies.")
# #print("Jack's proposed score:", JackScore4)
# if len(labelList2) == 1:
#     print("LlamaGuard Classification (should be safe!):", labelList4[0])
#     print("score:", score4)
    
# else:
#     print("LlamaGuard Classification (should be unsafe!):", labelList4[0])
#     print("category of violation (should be s10):", labelList4[1])
#     print("score:", score4)
# print()
