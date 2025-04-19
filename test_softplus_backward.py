import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds

# Define a module with the operator
class SoftplusBackwardModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.softplus_backward(out_grad, x, beta, threshold)

# Instantiate the model
mod = SoftplusBackwardModule()

# Define inputs
out_grad = None  # Unknown type\nx = torch.randn(3, requires_grad=True)\nbeta = torch.randn(3, requires_grad=True)\nthreshold = 1.0

args = (out_grad, x, beta, threshold,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
