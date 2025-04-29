import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SmoothL1Loss_OutModule(torch.nn.Module):
    def forward(self, x, target, reduction, beta, out):
        return torch.ops.aten.smooth_l1_loss.out(x, target, reduction, beta, out=out)

mod = Torch_Ops_Aten_SmoothL1Loss_OutModule()

x = torch.randn(3)
target = torch.randn(3)
reduction = 3
beta = 1.0
out = torch.empty(3)

args = (x, target, reduction, beta, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
