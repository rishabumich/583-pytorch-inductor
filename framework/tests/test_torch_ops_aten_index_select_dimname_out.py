import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexSelect_DimnameOutModule(torch.nn.Module):
    def forward(self, x, dim, index, out):
        return torch.ops.aten.index_select.dimname_out(x, dim, index, out=out)

mod = Torch_Ops_Aten_IndexSelect_DimnameOutModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
index = torch.randn(3)
out = torch.empty(3)

args = (x, dim, index, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
