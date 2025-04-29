import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HuberLossModule(torch.nn.Module):
    def forward(self, x, target, reduction, delta):
        return torch.ops.aten.huber_loss(x, target, reduction, delta)

mod = Torch_Ops_Aten_HuberLossModule()

x = torch.randn(3)
target = torch.randn(3)
reduction = 3
delta = 1.0

args = (x, target, reduction, delta,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
