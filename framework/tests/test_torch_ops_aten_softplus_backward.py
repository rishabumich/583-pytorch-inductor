import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SoftplusBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, beta, threshold):
        return torch.ops.aten.softplus_backward(grad_output, x, beta, threshold)

mod = Torch_Ops_Aten_SoftplusBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
beta = 1
threshold = 1

args = (grad_output, x, beta, threshold,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
