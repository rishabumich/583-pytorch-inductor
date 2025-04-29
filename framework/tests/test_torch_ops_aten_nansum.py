import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NansumModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, dtype):
        return torch.ops.aten.nansum(x, dim, keepdim, dtype)

mod = Torch_Ops_Aten_NansumModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
keepdim = True
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType?

args = (x, dim, keepdim, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
