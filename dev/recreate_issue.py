import torch

# Define a function that uses repeat_interleave and is compiled
@torch.compile()
def f(input, repeats):
    return torch.repeat_interleave(input, repeats, dim=0, output_size=3) + 1

# Input tensors on CUDA
input = torch.tensor([[1, 2], [3, 4]], device="cpu")
repeat = torch.tensor([1, 2], device="cpu")

# Run the function
output = f(input, repeat)
print("Output:\n", output)
