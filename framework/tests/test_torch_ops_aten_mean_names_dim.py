import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mean_NamesDimModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, dtype):
        return torch.ops.aten.mean.names_dim(x, dim, keepdim, dtype)

mod = Torch_Ops_Aten_Mean_NamesDimModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str[1]
keepdim = True
dtype = None  # Fallback for unknown type ScalarType?

args = (x, dim, keepdim, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
