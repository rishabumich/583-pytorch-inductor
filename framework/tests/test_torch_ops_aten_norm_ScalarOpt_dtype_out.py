import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_ScalaroptDtypeOutModule(torch.nn.Module):
    def forward(self, x, p, dtype, out):
        return torch.ops.aten.norm.ScalarOpt_dtype_out(x, p, dtype, out=out)

mod = Torch_Ops_Aten_Norm_ScalaroptDtypeOutModule()

x = torch.randn(3)
p = 1
dtype = None  # Fallback for unknown type ScalarType
out = torch.empty(3)

args = (x, p, dtype, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
