"""
Contains various implementations of scaled dot product attention.
See section 3.2.1 of Vaswani et al., 2018.
"""
import math
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

def normal_attention_torch(q, k, v):
    """
    Torch's implementation of scaled dot product attention in C++.
    """
    # with sdpa_kernel(backends=SDPBackend.MATH):
    #     start = time.perf_counter()
    #     output =  F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)
    #     end_time = time.perf_counter() - start
    #     return output, end_time
    start = time.perf_counter()
    output = scaled_dot_product_attention(q, k, v, is_causal=True)
    end_time = time.perf_counter() - start
    return output, end_time
    
    

def scaled_dot_product_attention(query, key, value, attn_mask=None,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value



def flash_attention_torch(q, k, v):
    """
    Torch's implementation of FlashAttention.
    """
    # Avoid UserWarning: All fused kernels requires query, key and value to be 4 dimensional
    if q.dim() == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
            
    with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        start = time.perf_counter()
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0) 
        end_time = time.perf_counter() - start
        return result, end_time
# refrenced from https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py v1 changelog
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L, # output buffer for softmax denominators
    M,  # output buffer for maxes
    Out, # output 
    # stride is for pointer arithmetic: 1 * stride gets you to the next element in the given axis (axis 0 is row, so have to go through full column if 2d)
    stride_head,    # step to get to next head
    stride_seq,     # step to get to next sequence within the head
    hidden_dim,     # step to get to next hidden dim
    seq_len,
    Br: tl.constexpr,  # Br in paper
    Bt: tl.constexpr,  # Bt in paper
    hidden: tl.constexpr,  # Hidden dim
):  
    start_m = tl.program_id(0)      # which portion of the sequence am I in? Which block of the queries is mine?
    off_hz = tl.program_id(1)       # which batch and head am I in? But really just which head?
    # initialize offsets (we grab a specific query embedding block and then iterate through k and v's in blocks)
    offs_br = start_m * Br + tl.arange(0, Br)      # get all of the rows of the query that I am responsible for
    offs_bt = tl.arange(0, Bt)                          # Identify the columns that I am responsible for
    offs_d = tl.arange(0, hidden)                     # get me all the hidden dims for the blocks

    off_q = off_hz * stride_head + offs_br[:, None] * stride_seq + offs_d[None, :] * hidden_dim
    #             ^^^                           ^^^                           ^^^
    #       get the right head    get all of the correct queries      get all of the hidden dims
    # same logic for K and V but using the columns for the correct k,v
    off_kv = off_hz * stride_head + offs_bt[:, None] * stride_seq + offs_d[None, :] * hidden_dim # nice 2d list
    # Initialize pointers to Q, K, V using the above offsets
    q_ptrs = Q + off_q  
    k_ptrs = K + off_kv
    v_ptrs = V + off_kv
    
    # create buffers for m_i, l_i, and accumulator.
    m_i = tl.zeros([Br], dtype=tl.float16) - float("inf") # maxes is negative infinity
    l_i = tl.zeros([Br], dtype=tl.float16) # no denominators yet
    acc = tl.zeros([Br, hidden], dtype=tl.float16) # zero out accumulator
    
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator

    for start_n in tl.range(0, (start_m + 1) * Br, Bt):
        # load k, v
        k = tl.load(k_ptrs + start_n * stride_seq)
        v = tl.load(v_ptrs + start_n * stride_seq)
        # Calculate qk and mask out future if causal
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk += tl.where(offs_br[:, None] >= (start_n + offs_bt[None, :]), 0, float("-inf")) # mask out future
        
        # Compute the softmax using the old max
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        
        # update the max and denominator
        m_i_new = tl.maximum(m_i, m_ij)
        # alpha can be multiplied to cancel the old max, and then subtract the new max
        alpha = tl.exp(m_i - m_i_new)
        # beta can be multiplied to cancel the current max in q and then subtract the new max
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # scale the softmax proabilities by new max and new denominator
        p = p * (beta / l_i_new)[:, None]
        p = p.to(v.dtype)

        # scale the output attention values by the new max and new denominator
        acc_scale_factor = l_i / l_i_new * alpha
        acc = acc * acc_scale_factor[:, None] + tl.dot(p, v)
        # update m_i and l_i to the new values
        acc = acc.to(v.dtype) # make sure the dtype is correct
        l_i = l_i_new.to(l_i.dtype)
        m_i = m_i_new.to(m_i.dtype)
    # write back l and m, and o
    l_ptrs = L + off_hz * seq_len + offs_br
    m_ptrs = M + off_hz * seq_len + offs_br
    out_ptrs = Out + off_q
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    tl.store(out_ptrs, acc)

def triton_forward(q, k, v):
    start = time.perf_counter()
    BLOCK = 64
    o = torch.empty_like(q)  # create space for output
    sm_scale = q.size(-1) ** -0.5

    # 2D launch grid of (r,c) where r = ceil(N/block size) and c = B*H
    # So, grid = (seqlen per block, batch-heads)
    # We are free to parallelize over any part of the query sequence (arbitrary query embedding attn output doesn't depend on other query embeddings)
    # We are free to parallelize over batch and heads
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
    
    # in paper, L and m are of size N; here, N-length L and m for each batch-head
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)    # Softmax denominators
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)    # maxes
    
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        L,
        m,
        o,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.shape[2],
        Br=BLOCK,
        Bt=BLOCK,
        hidden=q.shape[3],
    )
    end_time = time.perf_counter() - start
    return o, end_time

