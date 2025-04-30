import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Dist_OutModule(torch.nn.Module):
    def forward(self, x, other, p, out):
        return torch.ops.aten.dist.out(x, other, p, out=out)

mod = Torch_Ops_Aten_Dist_OutModule()

x = torch.randn(3)
other = torch.randn(3)
p = 1
out = torch.empty(3)

args = (x, other, p, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
