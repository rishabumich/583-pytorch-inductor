import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_ScalaroptDtypeModule(torch.nn.Module):
    def forward(self, x, p, dtype):
        return torch.ops.aten.norm.ScalarOpt_dtype(x, p, dtype)

mod = Torch_Ops_Aten_Norm_ScalaroptDtypeModule()

x = torch.randn(3)
p = 1
dtype = None  # Fallback for unknown type ScalarType

args = (x, p, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
