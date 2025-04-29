import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BinaryCrossEntropy_OutModule(torch.nn.Module):
    def forward(self, x, target, weight, reduction, out):
        return torch.ops.aten.binary_cross_entropy.out(x, target, weight, reduction, out=out)

mod = Torch_Ops_Aten_BinaryCrossEntropy_OutModule()

x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
reduction = 3
out = torch.empty(3)

args = (x, target, weight, reduction, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
