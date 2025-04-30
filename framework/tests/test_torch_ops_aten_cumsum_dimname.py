import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Cumsum_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, dtype):
        return torch.ops.aten.cumsum.dimname(x, dim, dtype)

mod = Torch_Ops_Aten_Cumsum_DimnameModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
dtype = None  # Fallback for unknown type ScalarType?

args = (x, dim, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
