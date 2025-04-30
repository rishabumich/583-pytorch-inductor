import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Kthvalue_DimnameModule(torch.nn.Module):
    def forward(self, x, k, dim, keepdim):
        return torch.ops.aten.kthvalue.dimname(x, k, dim, keepdim)

mod = Torch_Ops_Aten_Kthvalue_DimnameModule()

x = torch.randn(3)
k = 3
dim = None  # Fallback for unknown type str
keepdim = True

args = (x, k, dim, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
