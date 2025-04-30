import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sum_IntlistOutModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, dtype, out):
        return torch.ops.aten.sum.IntList_out(x, dim, keepdim, dtype, out=out)

mod = Torch_Ops_Aten_Sum_IntlistOutModule()

x = torch.randn(3)
dim = 3
keepdim = True
dtype = None  # Fallback for unknown type ScalarType?
out = torch.empty(3)

args = (x, dim, keepdim, dtype, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
