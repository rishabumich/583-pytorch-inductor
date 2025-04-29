import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UniqueDimModule(torch.nn.Module):
    def forward(self, x, dim, sorted, return_inverse, return_counts):
        return torch.ops.aten.unique_dim(x, dim, sorted, return_inverse, return_counts)

mod = Torch_Ops_Aten_UniqueDimModule()

x = torch.randn(3)
dim = 3
sorted = True
return_inverse = True
return_counts = True

args = (x, dim, sorted, return_inverse, return_counts,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
