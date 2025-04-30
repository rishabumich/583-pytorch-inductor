import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sort_DimnameStableModule(torch.nn.Module):
    def forward(self, x, stable, dim, descending):
        return torch.ops.aten.sort.dimname_stable(x, stable, dim, descending)

mod = Torch_Ops_Aten_Sort_DimnameStableModule()

x = torch.randn(3)
stable = True
dim = None  # Fallback for unknown type str
descending = True

args = (x, stable, dim, descending,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
