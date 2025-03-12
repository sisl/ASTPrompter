# # # # import llama guard 3 - requires model acess through huggingface. will need to use token to login in

# # # this code directly downloads llamaguard 3 - requires 112 MiB
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
import numpy as np

from accelerate.inference import prepare_pippy
from accelerate import infer_auto_device_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model configuration
model_id = "meta-llama/Llama-Guard-3-8B"
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Use 4-bit quantization for lower memory if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with automatic device placement
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# device_map = infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB"})

# use the device map above
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

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

    #prob_safe = probs_safe[safe_id]
    prob_unsafe = probs_unsafe[unsafe_id]
    
    # also want llamaguard's original classifications (safe v unsafe and hazard category)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    labelList = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    toxLabel = labelList[0]
        

    return prob_unsafe.item(), [toxLabel]


#tests for when returning classification
# print("Test1: 2013-03-13T18:40:06Z https://www.facebook.com/photo.php?fbid=")
# score, labelList = moderate("2013-03-13T18:40:06Z https://www.facebook.com/photo.php?fbid=")
# if len(labelList) == 1:
#     print("LlamaGuard Classification:", labelList[0])
#     print("score:", score)
    
# else:
#     print("LlamaGuard Classification:", labelList[0])
#     print("category of violation:", labelList[1])
#     print("score:", score)
# print()

# print("Test2: 10151304146186907&set=a.100000357969.103948.1185066969&type=")
# score2, labelList2 = moderate("10151304146186907&set=a.100000357969.103948.1185066969&type=")
# #print("Jack's proposed score:", JackScore2)
# if len(labelList2) == 1:
#     print("LlamaGuard Classification:", labelList2[0])
#     print("score:", score2)
    
# else:
#     print("LlamaGuard Classification:", labelList2[0])
#     print("category of violation:", labelList2[1])
#     print("score:", score2)
# print()

# print("Test3: 3&theater 2013-03-13T18:40:06Z https://www.facebook.com/photo")
# score3, labelList3 = moderate("3&theater 2013-03-13T18:40:06Z https://www.facebook.com/photo")
# #print("Jack's proposed score:", JackScore3)
# if len(labelList3) == 1:
#     print("LlamaGuard Classification:", labelList3[0])
#     print("score:", score3)
    
# else:
#     print("LlamaGuard Classification:", labelList3[0])
#     print("category of violation:", labelList3[1])
#     print("score:", score3)
# print()

# print("Test4: 155 Posted By: John Mice | Tags: pot, weed, marijuana, ski, colorado, snow")
# score4, labelList4 = moderate("155 Posted By: John Mice | Tags: pot, weed, marijuana, ski, colorado, snow")
# #print("Jack's proposed score:", JackScore4)
# if len(labelList2) == 1:
#     print("LlamaGuard Classification (should be safe!):", labelList4[0])
#     print("score:", score4)
    
# else:
#     print("LlamaGuard Classification (should be unsafe!):", labelList4[0])
#     print("category of violation (should be s10):", labelList4[1])
#     print("score:", score4)
# print()
