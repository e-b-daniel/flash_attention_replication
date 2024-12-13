import time
from typing import Callable
import torch
from attention import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def benchmark(atten_fn: Callable, batch_size, num_heads, seq_len, hidden_dim, n=100):
    avg_time = 0
    for _ in range(n):
        q = torch.rand(batch_size,num_heads, seq_len, hidden_dim, dtype=torch.float16, device=device, requires_grad=False)        
        k = torch.rand(batch_size, num_heads,seq_len, hidden_dim, dtype=torch.float16,device=device, requires_grad=False)
        v = torch.rand(batch_size, num_heads,seq_len, hidden_dim, dtype=torch.float16, device=device, requires_grad=False)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _, time_taken = atten_fn(q=q, k=k, v=v)
        avg_time += time_taken
        del q, k, v
    return avg_time / n

def check_accuracy(unverified_atten: Callable, batch_size, num_heads, seq_len, hidden_dim):
    q = torch.rand(batch_size,num_heads, seq_len, hidden_dim, dtype=torch.float16, device=device, requires_grad=False)        
    k = torch.rand(batch_size, num_heads,seq_len, hidden_dim, dtype=torch.float16,device=device, requires_grad=False)
    v = torch.rand(batch_size, num_heads,seq_len, hidden_dim, dtype=torch.float16, device=device, requires_grad=False)

    ground_truth, _ = normal_attention_torch(q,k,v)
    test_implementation, _ = unverified_atten(q,k,v)
    if torch.allclose(ground_truth, test_implementation):
        print('Implementation is correct')
    else:
        print('Implementation is incorrect')
        print(torch.nn.functional.mse_loss(ground_truth, test_implementation))
        bools = torch.isclose(ground_truth, test_implementation)
        percent_correct = torch.sum(bools) / bools.numel() * 100
        print(f'Only {percent_correct}% of output values matched the ground truth implementation.')

@torch.no_grad()
def main():
    batch_size = 50
    seq_len = 512
    hidden_dim = 64
    num_heads = 12

    print("Warmup")
    print("\tNormal Attention (Torch / C++):   \t", benchmark(normal_attention_torch, batch_size, num_heads, seq_len, hidden_dim,n=3))
    print("\tFlash Attention (Ours / Triton): \t", benchmark(triton_forward, batch_size, num_heads, seq_len, hidden_dim, n=3))
    print("\tFlash Attention (Torch / C++):   \t", benchmark(flash_attention_torch, batch_size,num_heads,  seq_len, hidden_dim, n=3))
    print()
    
    # check all the accuracies
    print("Checking Accuracy")
    print("\tNormal Attention (Torch / C++):")
    check_accuracy(normal_attention_torch, batch_size, num_heads, seq_len, hidden_dim)
    print("\tFlash Attention (Ours / Triton):")
    check_accuracy(triton_forward, batch_size, num_heads, seq_len, hidden_dim)
    print("\tFlash Attention (Torch / C++):")
    check_accuracy(flash_attention_torch, batch_size, num_heads, seq_len, hidden_dim)

    print("Benchmark")
    print("\tNormal Attention (Torch / C++):   \t", benchmark(normal_attention_torch, batch_size, num_heads, seq_len, hidden_dim))
    print("\tFlash Attention (Ours / Triton): \t", benchmark(triton_forward, batch_size, num_heads, seq_len, hidden_dim))
    print("\tFlash Attention (Torch / C++):   \t", benchmark(flash_attention_torch, batch_size, num_heads, seq_len, hidden_dim))
    return

if __name__ == '__main__':
    main()

