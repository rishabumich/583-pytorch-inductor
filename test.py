import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds

# Define a module with a split operation
class SplitMod(torch.nn.Module):
    def forward(self, x):
        return torch.split(x, [1, 1], dim=0)

# Instantiate the model
mod = SplitMod()

# Input tensor (ensure it's on the correct device)
x = torch.tensor([[1, 2], [3, 4]])
args = (x,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
