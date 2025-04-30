import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_ScalaroptDimDtypeModule(torch.nn.Module):
    def forward(self, x, p, dim, keepdim, dtype):
        return torch.ops.aten.norm.ScalarOpt_dim_dtype(x, p, dim, keepdim, dtype)

mod = Torch_Ops_Aten_Norm_ScalaroptDimDtypeModule()

x = torch.randn(3)
p = 1
dim = 3
keepdim = True
dtype = None  # Fallback for unknown type ScalarType

args = (x, p, dim, keepdim, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
