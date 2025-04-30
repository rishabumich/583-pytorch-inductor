import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ScatterReduce_TwoModule(torch.nn.Module):
    def forward(self, x, dim, index, src, reduce, include_self):
        return torch.ops.aten.scatter_reduce_.two(x, dim, index, src, reduce, include_self)

mod = Torch_Ops_Aten_ScatterReduce_TwoModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
src = torch.randn(3)
reduce = None  # Fallback for unknown type str
include_self = True

args = (x, dim, index, src, reduce, include_self,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
