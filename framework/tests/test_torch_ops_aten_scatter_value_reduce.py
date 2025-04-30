import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Scatter_ValueReduceModule(torch.nn.Module):
    def forward(self, x, dim, index, value, reduce):
        return torch.ops.aten.scatter.value_reduce(x, dim, index, value, reduce)

mod = Torch_Ops_Aten_Scatter_ValueReduceModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
value = 1
reduce = None  # Fallback for unknown type str

args = (x, dim, index, value, reduce,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
