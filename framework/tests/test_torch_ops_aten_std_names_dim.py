import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Std_NamesDimModule(torch.nn.Module):
    def forward(self, x, dim, unbiased, keepdim):
        return torch.ops.aten.std.names_dim(x, dim, unbiased, keepdim)

mod = Torch_Ops_Aten_Std_NamesDimModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str[1]
unbiased = True
keepdim = True

args = (x, dim, unbiased, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
