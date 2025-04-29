import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Prod_DimDimnameModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, dtype):
        return torch.ops.aten.prod.dim_Dimname(x, dim, keepdim, dtype)

mod = Torch_Ops_Aten_Prod_DimDimnameModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
keepdim = True
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType?

args = (x, dim, keepdim, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
