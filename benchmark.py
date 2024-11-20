from typing import Callable
import torch
from timeit import timeit
from functools import partial
from attention import *

def benchmark(atten_fn: Callable, batch_size, seq_len, hidden_dim, causal_mask=False, n=10):
    q = torch.rand(batch_size, seq_len, hidden_dim)
    k = torch.rand(batch_size, seq_len, hidden_dim)
    v = torch.rand(batch_size, seq_len, hidden_dim)
    mask = torch.tril(torch.ones((seq_len, seq_len))) if causal_mask else None
    avg_time = timeit(lambda:atten_fn(q=q, k=k, v=v, mask=mask), number=n) / n
    return avg_time
  
def main():
    batch_size = 256
    seq_len = 512
    hidden_dim = 2048
    
    print()
    print("\tNormal Attention (Ours / Python): \t", benchmark(normal_attention_ours, batch_size, seq_len, hidden_dim))
    print("\tNormal Attention (Torch / C++):   \t", benchmark(normal_attention_torch, batch_size, seq_len, hidden_dim))
    # print("Flash Attention (Ours / Triton): \t", benchmark(flash_attention_ours, batch_size, seq_len, hidden_dim))
    print("\tFlash Attention (Torch / C++):   \t", benchmark(flash_attention_torch, batch_size, seq_len, hidden_dim))
    print()

if __name__ == '__main__':
    main()