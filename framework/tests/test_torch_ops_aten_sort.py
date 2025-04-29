import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SortModule(torch.nn.Module):
    def forward(self, x, dim, descending):
        return torch.ops.aten.sort(x, dim, descending)

mod = Torch_Ops_Aten_SortModule()

x = torch.randn(3)
dim = 3
descending = True

args = (x, dim, descending,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
