import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UniqueDim_OutModule(torch.nn.Module):
    def forward(self, x, dim, sorted, return_inverse, return_counts, out0, out1, out2):
        return torch.ops.aten.unique_dim.out(x, dim, sorted, return_inverse, return_counts, out0, out1, out2)

mod = Torch_Ops_Aten_UniqueDim_OutModule()

x = torch.randn(3)
dim = 3
sorted = True
return_inverse = True
return_counts = True
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (x, dim, sorted, return_inverse, return_counts, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
