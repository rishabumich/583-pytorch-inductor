import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexCopy_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, index, source):
        return torch.ops.aten.index_copy.dimname(x, dim, index, source)

mod = Torch_Ops_Aten_IndexCopy_DimnameModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
index = torch.randn(3)
source = torch.randn(3)

args = (x, dim, index, source,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
