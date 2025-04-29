import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BinaryCrossEntropyWithLogitsModule(torch.nn.Module):
    def forward(self, x, target, weight, pos_weight, reduction):
        return torch.ops.aten.binary_cross_entropy_with_logits(x, target, weight, pos_weight, reduction)

mod = Torch_Ops_Aten_BinaryCrossEntropyWithLogitsModule()

x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
pos_weight = torch.randn(3)
reduction = 3

args = (x, target, weight, pos_weight, reduction,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
