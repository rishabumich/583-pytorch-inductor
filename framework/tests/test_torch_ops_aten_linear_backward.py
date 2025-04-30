import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinearBackwardModule(torch.nn.Module):
    def forward(self, x, grad_output, weight, output_mask):
        return torch.ops.aten.linear_backward(x, grad_output, weight, output_mask)

mod = Torch_Ops_Aten_LinearBackwardModule()

x = torch.randn(3)
grad_output = torch.randn(3)
weight = torch.randn(3)
output_mask = True

args = (x, grad_output, weight, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
