import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Min_DimMinModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, min, min_indices):
        return torch.ops.aten.min.dim_min(x, dim, keepdim, min, min_indices)

mod = Torch_Ops_Aten_Min_DimMinModule()

x = torch.randn(3)
dim = 3
keepdim = True
min = torch.randn(3)
min_indices = torch.randn(3)

args = (x, dim, keepdim, min, min_indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
