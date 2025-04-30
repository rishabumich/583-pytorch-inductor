import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexAdd_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, index, source, alpha):
        return torch.ops.aten.index_add.dimname(x, dim, index, source, alpha)

mod = Torch_Ops_Aten_IndexAdd_DimnameModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
index = torch.randn(3)
source = torch.randn(3)
alpha = 1

args = (x, dim, index, source, alpha,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
