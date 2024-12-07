from typing import Callable
import torch
from timeit import timeit
from functools import partial
from attention import *
from flash import *

def benchmark(atten_fn: Callable, batch_size, seq_len, hidden_dim, causal_mask=False, n=10):
    q = torch.rand(batch_size, seq_len, hidden_dim)[0]
    k = torch.rand(batch_size, seq_len, hidden_dim)[0]
    v = torch.rand(batch_size, seq_len, hidden_dim)[0]
    print('trying')
    flash_attention_python(q=q, k=k, v=v)
    print('done')
    # mask = torch.tril(torch.ones((seq_len, seq_len))) if causal_mask else None
    # avg_time = timeit(lambda:atten_fn(q=q, k=k, v=v), number=n) / n
    # return avg_time

def check_accuracy(unverified_atten: Callable):
    seq_len, hidden_dim = 64, 16
    q = torch.rand(seq_len, hidden_dim)
    k = torch.rand(seq_len, hidden_dim)
    v = torch.rand(seq_len, hidden_dim)
    print('t2')
    ground_truth = normal_attention_torch(q,k,v)
    print('t3')
    test_implementation = unverified_atten(q,k,v)
    print('t4')
    if torch.allclose(ground_truth, test_implementation):
        print('Implementation is correct')
    else:
        print('Implementation is incorrect')
        print(torch.nn.functional.mse_loss(ground_truth, test_implementation))
        bools = torch.isclose(ground_truth, test_implementation)
        percent_correct = torch.sum(bools) / sum(bools.size())
        print(f'Only {percent_correct}% of output values matched the ground truth implementation.')
    # mask = torch.tril(torch.ones((seq_len, seq_len))) if causal_mask else None
    # avg_time = timeit(lambda:atten_fn(q=q, k=k, v=v), number=n) / n
    # return avg_time
  
def main():
    print('yo')
    batch_size = 256
    seq_len = 512
    hidden_dim = 2048
    check_accuracy(flash_attention_python)
    # print()
    # print("\tNormal Attention (Ours / Python): \t", benchmark(normal_attention_ours, batch_size, seq_len, hidden_dim))
    # print("\tNormal Attention (Torch / C++):   \t", benchmark(normal_attention_torch, batch_size, seq_len, hidden_dim))
    # print("Flash Attention (Ours / Triton): \t", benchmark(flash_attention_python, batch_size, seq_len, hidden_dim))
    # print("\tFlash Attention (Torch / C++):   \t", benchmark(flash_attention_torch, batch_size, seq_len, hidden_dim))
    # print()

if __name__ == '__main__':
    main()

