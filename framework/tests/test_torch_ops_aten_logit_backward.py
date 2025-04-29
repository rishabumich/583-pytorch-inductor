import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogitBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, eps):
        return torch.ops.aten.logit_backward(grad_output, x, eps)

mod = Torch_Ops_Aten_LogitBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
eps = torch.tensor(0)  # Fallback for unknown type float?

args = (grad_output, x, eps,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
