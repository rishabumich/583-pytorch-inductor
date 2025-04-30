import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Norm_NamesOutModule(torch.nn.Module):
    def forward(self, x, p, dim, keepdim, out):
        return torch.ops.aten.norm.names_out(x, p, dim, keepdim, out=out)

mod = Torch_Ops_Aten_Norm_NamesOutModule()

x = torch.randn(3)
p = 1
dim = None  # Fallback for unknown type str[1]
keepdim = True
out = torch.empty(3)

args = (x, p, dim, keepdim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
