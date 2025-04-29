import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Any_DimnameOutModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, out):
        return torch.ops.aten.any.dimname_out(x, dim, keepdim, out=out)

mod = Torch_Ops_Aten_Any_DimnameOutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
keepdim = True
out = torch.empty(3)

args = (x, dim, keepdim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
