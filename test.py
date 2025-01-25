import torch
from pynvml import *

nvmlInit()

free_memory = []
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print((torch.cuda.get_device_properties(i).total_memory, torch.cuda.memory_allocated(i), torch.cuda.memory_reserved(i)))
    h = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    free_memory.append((torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i), i))
_, best_gpu = max(free_memory)

print(f"Free Memtory: {free_memory}")
print(f"Best GPU: {best_gpu}")
