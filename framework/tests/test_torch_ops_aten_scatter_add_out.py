import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ScatterAdd_OutModule(torch.nn.Module):
    def forward(self, x, dim, index, src, out):
        return torch.ops.aten.scatter_add.out(x, dim, index, src, out=out)

mod = Torch_Ops_Aten_ScatterAdd_OutModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
src = torch.randn(3)
out = torch.empty(3)

args = (x, dim, index, src, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
