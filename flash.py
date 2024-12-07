import torch
import math
import triton
import triton.language as tl

M = 100000000000000

def flash_attention_python(q, k, v):
    Q, K, V = q, k, v
    # get block size
    N, d = Q.size()
    Bc = math.ceil(M / (4*d))
    Br = min(Bc, d)

    # initialize output / tmp variables
    O = torch.zeros((N, d))
    l = torch.zeros(N)
    m = torch.full((N,), float('-inf'))

    # divide into blocks that fit onto SRAM
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)

    # outer loop over columns
    for j in range(Tc):
        Kj, Vj = K[j * Bc : j * Bc + Bc], V[j * Bc : j * Bc + Bc]
        # inner loop over rows
        for i in range(Tr):
            Qi, Oi, li, mi = Q[Br * i : Br * i + Br], O[Br * i : Br * i + Br], l[i * Br : i * Br + Br], m[i * Br : i * Br + Br]
            Sij = Qi @ Kj.T
            mij = torch.max(Sij, axis=1).values
            Pij = torch.exp(Sij - mij.unsqueeze(1))
            lij = torch.sum(Pij, axis=1)
            minew = torch.maximum(mi, mij)
            linew = torch.exp(mi - minew) * li + torch.exp(mij - minew) * lij
            #                         (Br, Br)                   (Br, Br)         (D, D) = (Br)
            O[Br * i : Br * i + Br] = (torch.diag(1 / linew)) @ ((torch.diag(li) * torch.exp(mi - minew)) @ Oi + torch.exp(mij - minew).unsqueeze(1) * Pij @ Vj)
            # O[Br * i : Br * i + Br] = (torch.diag(1 / linew)) @ ((li * torch.exp(mi - minew)) @ Oi + torch.exp(mij - minew).unsqueeze(1) * Pij @ Vj)

            l[Br * i : Br * i + Br] = linew
            m[Br * i : Br * i + Br] = minew
    return O

def n(x, name):
    if math.isnan(x):
        print(f'{name} is not a number')

def flash_attention_triton(query, key, value, mask=None):
    """
    Driver function.
    """
    pass