import torch
import triton
import triton.language as tl

def flash_attention_triton(query, key, value, mask=None):
  pass