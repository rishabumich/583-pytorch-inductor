import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NllLoss2DForward_OutputModule(torch.nn.Module):
    def forward(self, x, target, weight, reduction, ignore_index, output, total_weight):
        return torch.ops.aten.nll_loss2d_forward.output(x, target, weight, reduction, ignore_index, output, total_weight)

mod = Torch_Ops_Aten_NllLoss2DForward_OutputModule()

x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
reduction = 3
ignore_index = torch.tensor(0)  # Fallback for unknown type SymInt
output = torch.randn(3)
total_weight = torch.randn(3)

args = (x, target, weight, reduction, ignore_index, output, total_weight,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
