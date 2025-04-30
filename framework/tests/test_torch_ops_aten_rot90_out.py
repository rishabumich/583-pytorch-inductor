import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Rot90_OutModule(torch.nn.Module):
    def forward(self, x, k, dims, out):
        return torch.ops.aten.rot90.out(x, k, dims, out=out)

mod = Torch_Ops_Aten_Rot90_OutModule()

x = torch.randn(3)
k = 3
dims = 3
out = torch.empty(3)

args = (x, k, dims, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
