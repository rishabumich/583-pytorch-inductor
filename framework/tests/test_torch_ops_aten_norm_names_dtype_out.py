import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_NamesDtypeOutModule(torch.nn.Module):
    def forward(self, x, p, dim, keepdim, dtype, out):
        return torch.ops.aten.norm.names_dtype_out(x, p, dim, keepdim, dtype, out=out)

mod = Torch_Ops_Aten_Norm_NamesDtypeOutModule()

x = torch.randn(3)
p = torch.tensor(0)  # Fallback for unknown type Scalar?
dim = torch.tensor(0)  # Fallback for unknown type str[1]
keepdim = True
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType
out = torch.empty(3)

args = (x, p, dim, keepdim, dtype, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
