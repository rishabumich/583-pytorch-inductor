import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_DtypeOutModule(torch.nn.Module):
    def forward(self, x, p, dim, keepdim, dtype, out):
        return torch.ops.aten.norm.dtype_out(x, p, dim, keepdim, dtype, out=out)

mod = Torch_Ops_Aten_Norm_DtypeOutModule()

x = torch.randn(3)
p = 1
dim = 3
keepdim = True
dtype = None  # Fallback for unknown type ScalarType
out = torch.empty(3)

args = (x, p, dim, keepdim, dtype, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
