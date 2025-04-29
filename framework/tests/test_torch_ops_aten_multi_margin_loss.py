import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MultiMarginLossModule(torch.nn.Module):
    def forward(self, x, target, p, margin, weight, reduction):
        return torch.ops.aten.multi_margin_loss(x, target, p, margin, weight, reduction)

mod = Torch_Ops_Aten_MultiMarginLossModule()

x = torch.randn(3)
target = torch.randn(3)
p = torch.tensor(0)  # Fallback for unknown type Scalar
margin = torch.tensor(0)  # Fallback for unknown type Scalar
weight = torch.randn(3)
reduction = 3

args = (x, target, p, margin, weight, reduction,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
