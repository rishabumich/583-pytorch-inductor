import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Renorm_OutModule(torch.nn.Module):
    def forward(self, x, p, dim, maxnorm, out):
        return torch.ops.aten.renorm.out(x, p, dim, maxnorm, out=out)

mod = Torch_Ops_Aten_Renorm_OutModule()

x = torch.randn(3)
p = 1
dim = 3
maxnorm = 1
out = torch.empty(3)

args = (x, p, dim, maxnorm, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
