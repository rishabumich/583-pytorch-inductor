import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sort_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, descending):
        return torch.ops.aten.sort.dimname(x, dim, descending)

mod = Torch_Ops_Aten_Sort_DimnameModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
descending = True

args = (x, dim, descending,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
