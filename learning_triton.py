import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel( x_ptr,
                y_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr):
    # figure out what part of the inputs i'm operating on
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start * tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements     # make sure i don't try to load in parts of input that don't exist

    # load from DRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # perform computation
    result = x + y

    # write result to DRAM
    tl.store(output_ptr, result, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # allocate space for output
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (tl.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
