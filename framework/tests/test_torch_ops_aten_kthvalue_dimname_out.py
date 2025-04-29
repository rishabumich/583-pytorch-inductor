import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Kthvalue_DimnameOutModule(torch.nn.Module):
    def forward(self, x, k, dim, keepdim, values, indices):
        return torch.ops.aten.kthvalue.dimname_out(x, k, dim, keepdim, values, indices)

mod = Torch_Ops_Aten_Kthvalue_DimnameOutModule()

x = torch.randn(3)
k = 3
dim = torch.tensor(0)  # Fallback for unknown type str
keepdim = True
values = torch.randn(3)
indices = torch.randn(3)

args = (x, k, dim, keepdim, values, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
