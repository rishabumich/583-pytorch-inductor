import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_NamesScalaroptDimModule(torch.nn.Module):
    def forward(self, x, p, dim, keepdim):
        return torch.ops.aten.norm.names_ScalarOpt_dim(x, p, dim, keepdim)

mod = Torch_Ops_Aten_Norm_NamesScalaroptDimModule()

x = torch.randn(3)
p = 1
dim = None  # Fallback for unknown type str[1]
keepdim = True

args = (x, p, dim, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
