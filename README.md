# flash_attention_replication

This repo contains code for various implementations of attention, alongside testing code to compare their efficiencies., In particular we prove the following implementations of attention:

1. A naive implementation in Python that simply translates the equation for Scaled Dot-Product Attention from Section 3.2.1 from [Vaswani et al., 2018](https://arxiv.org/pdf/1706.03762#page=4). 
2. An optimized implementation in C++ translates the scaled dot-product attention equation, provided by PyTorch.
3. Our implementation of Flash Attention in Triton.
4. PyTorch's implementation of Flash Attention.

These implementations can be found in `attention.py`, under `normal_attention_ours`, `normal_attention_torch`, `flash_attention_ours`, and `flash_attention_torch`, respectively.