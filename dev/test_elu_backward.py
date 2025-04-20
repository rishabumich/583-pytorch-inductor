import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table  # use _decomp as of current PyTorch nightly builds

# Define a module with the operator
class EluBackwardModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result)

# Instantiate the model
mod = EluBackwardModule()

# Define inputs
grad_output = None  # Unknown type\nalpha = torch.randn(3, requires_grad=True)\nscale = 1.0\ninput_scale = 1.0\nis_result = 1.0\nself_or_result = True

args = (grad_output, alpha, scale, input_scale, is_result, self_or_result,)

# Export graph before decomposition
ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

# Apply decompositions
ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
