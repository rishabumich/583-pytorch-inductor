import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NllLossModule(torch.nn.Module):
    def forward(self, x, target, weight, reduction, ignore_index):
        return torch.ops.aten.nll_loss(x, target, weight, reduction, ignore_index)

mod = Torch_Ops_Aten_NllLossModule()

x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
reduction = 3
ignore_index = None  # Fallback for unknown type SymInt

args = (x, target, weight, reduction, ignore_index,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
