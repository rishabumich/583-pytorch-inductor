import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Scatter_ValueReduceOutModule(torch.nn.Module):
    def forward(self, x, dim, index, value, reduce, out):
        return torch.ops.aten.scatter.value_reduce_out(x, dim, index, value, reduce, out=out)

mod = Torch_Ops_Aten_Scatter_ValueReduceOutModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
value = 1
reduce = None  # Fallback for unknown type str
out = torch.empty(3)

args = (x, dim, index, value, reduce, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
