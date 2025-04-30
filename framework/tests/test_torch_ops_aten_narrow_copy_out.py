import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NarrowCopy_OutModule(torch.nn.Module):
    def forward(self, x, dim, start, length, out):
        return torch.ops.aten.narrow_copy.out(x, dim, start, length, out=out)

mod = Torch_Ops_Aten_NarrowCopy_OutModule()

x = torch.randn(3)
dim = 3
start = None  # Fallback for unknown type SymInt
length = None  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (x, dim, start, length, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
