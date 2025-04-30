import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Scatter_ReduceOutModule(torch.nn.Module):
    def forward(self, x, dim, index, src, reduce, out):
        return torch.ops.aten.scatter.reduce_out(x, dim, index, src, reduce, out=out)

mod = Torch_Ops_Aten_Scatter_ReduceOutModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
src = torch.randn(3)
reduce = None  # Fallback for unknown type str
out = torch.empty(3)

args = (x, dim, index, src, reduce, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
