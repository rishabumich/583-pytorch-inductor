import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Max_DimMaxModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, max, max_values):
        return torch.ops.aten.max.dim_max(x, dim, keepdim, max, max_values)

mod = Torch_Ops_Aten_Max_DimMaxModule()

x = torch.randn(3)
dim = 3
keepdim = True
max = torch.randn(3)
max_values = torch.randn(3)

args = (x, dim, keepdim, max, max_values,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
