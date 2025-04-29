import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ScatterAdd_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, index, src):
        return torch.ops.aten.scatter_add.dimname(x, dim, index, src)

mod = Torch_Ops_Aten_ScatterAdd_DimnameModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
index = torch.randn(3)
src = torch.randn(3)

args = (x, dim, index, src,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
