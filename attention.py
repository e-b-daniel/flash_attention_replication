"""
Contains various implementations of scaled dot product attention.
See section 3.2.1 of Vaswani et al., 2018.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash import flash_attention_triton

def normal_attention_ours(q, k, v, mask=None):
    """
    Our naive implementation of scaled dot product attention in Python.
    """
    logits = q @ k.transpose(-2, -1)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))
    scaled_logits = logits / (q.size(-1) ** 0.5)
    scores = F.softmax(scaled_logits, dim=-1)
    out = scores @ v
    return out

def normal_attention_torch(q, k, v, mask=None):
    """
    Torch's implementation of scaled dot product attention in C++.
    """
    with sdpa_kernel(backends=SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

def flash_attention_ours(q, k, v, mask=None):
    """
    Our implementation of FlashAttention in Triton.
    """
    return flash_attention_triton(q, k, v, mask)

def flash_attention_torch(q, k, v, mask=None):
    """
    Torch's implementation of FlashAttention.
    """
    # Avoid UserWarning: All fused kernels requires query, key and value to be 4 dimensional
    if q.dim() == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
            
    with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

def unspecified_attention_torch(q,k,v):
    F.scaled_dot_product_attention(q, k, v)