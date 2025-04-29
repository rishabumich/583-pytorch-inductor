import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds
import sys

# Redirect all stdout to a file
sys.stdout = open("output_elu_backward.txt", "w")

# Define a module with the operator
class EluBackwardModule(torch.nn.Module):
    def forward(self, grad_output, alpha, scale, input_scale, is_result, self_or_result):
        return torch.ops.aten.elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result)

# Instantiate the model
mod = EluBackwardModule()

# Define inputs
grad_output = torch.randn(3, requires_grad=True)
alpha = 1.0
scale = 1.0
input_scale = 1.0
is_result = True
self_or_result = torch.randn(3, requires_grad=True)

args = (grad_output, alpha, scale, input_scale, is_result, self_or_result,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
