import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Diagonal_DimnameModule(torch.nn.Module):
    def forward(self, x, outdim, dim1, dim2, offset):
        return torch.ops.aten.diagonal.Dimname(x, outdim, dim1, dim2, offset)

mod = Torch_Ops_Aten_Diagonal_DimnameModule()

x = torch.randn(3)
outdim = None  # Fallback for unknown type str
dim1 = None  # Fallback for unknown type str
dim2 = None  # Fallback for unknown type str
offset = 3

args = (x, outdim, dim1, dim2, offset,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
