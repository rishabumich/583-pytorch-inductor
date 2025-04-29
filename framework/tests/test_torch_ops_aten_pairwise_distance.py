import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PairwiseDistanceModule(torch.nn.Module):
    def forward(self, x1, x2, p, eps, keepdim):
        return torch.ops.aten.pairwise_distance(x1, x2, p, eps, keepdim)

mod = Torch_Ops_Aten_PairwiseDistanceModule()

x1 = torch.randn(3)
x2 = torch.randn(3)
p = 1.0
eps = 1.0
keepdim = True

args = (x1, x2, p, eps, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
