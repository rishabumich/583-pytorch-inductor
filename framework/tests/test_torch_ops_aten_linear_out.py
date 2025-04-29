import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Linear_OutModule(torch.nn.Module):
    def forward(self, input, weight, bias, out):
        return torch.ops.aten.linear.out(input, weight, bias, out=out)

mod = Torch_Ops_Aten_Linear_OutModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
out = torch.empty(3)

args = (input, weight, bias, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
