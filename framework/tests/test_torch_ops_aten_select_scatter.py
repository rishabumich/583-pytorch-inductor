import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SelectScatterModule(torch.nn.Module):
    def forward(self, x, src, dim, index):
        return torch.ops.aten.select_scatter(x, src, dim, index)

mod = Torch_Ops_Aten_SelectScatterModule()

x = torch.randn(3)
src = torch.randn(3)
dim = 3
index = None  # Fallback for unknown type SymInt

args = (x, src, dim, index,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
