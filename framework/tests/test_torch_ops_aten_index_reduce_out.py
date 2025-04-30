import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexReduce_OutModule(torch.nn.Module):
    def forward(self, x, dim, index, source, reduce, include_self, out):
        return torch.ops.aten.index_reduce.out(x, dim, index, source, reduce, include_self, out=out)

mod = Torch_Ops_Aten_IndexReduce_OutModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
source = torch.randn(3)
reduce = None  # Fallback for unknown type str
include_self = True
out = torch.empty(3)

args = (x, dim, index, source, reduce, include_self, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
