import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Select_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, index):
        return torch.ops.aten.select.Dimname(x, dim, index)

mod = Torch_Ops_Aten_Select_DimnameModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
index = 3

args = (x, dim, index,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
