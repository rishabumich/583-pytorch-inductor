import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Min_NamesDimModule(torch.nn.Module):
    def forward(self, x, dim, keepdim):
        return torch.ops.aten.min.names_dim(x, dim, keepdim)

mod = Torch_Ops_Aten_Min_NamesDimModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
keepdim = True

args = (x, dim, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
