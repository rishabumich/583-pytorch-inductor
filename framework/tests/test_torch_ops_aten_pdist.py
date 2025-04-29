import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PdistModule(torch.nn.Module):
    def forward(self, x, p):
        return torch.ops.aten.pdist(x, p)

mod = Torch_Ops_Aten_PdistModule()

x = torch.randn(3)
p = 1.0

args = (x, p,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
