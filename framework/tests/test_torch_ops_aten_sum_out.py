import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sum_OutModule(torch.nn.Module):
    def forward(self, x, dtype, out):
        return torch.ops.aten.sum.out(x, dtype, out=out)

mod = Torch_Ops_Aten_Sum_OutModule()

x = torch.randn(3)
dtype = None  # Fallback for unknown type ScalarType?
out = torch.empty(3)

args = (x, dtype, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
