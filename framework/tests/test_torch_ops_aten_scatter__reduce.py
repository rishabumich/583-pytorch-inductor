import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Scatter_ReduceModule(torch.nn.Module):
    def forward(self, x, dim, index, src, reduce):
        return torch.ops.aten.scatter_.reduce(x, dim, index, src, reduce)

mod = Torch_Ops_Aten_Scatter_ReduceModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
src = torch.randn(3)
reduce = torch.tensor(0)  # Fallback for unknown type str

args = (x, dim, index, src, reduce,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
