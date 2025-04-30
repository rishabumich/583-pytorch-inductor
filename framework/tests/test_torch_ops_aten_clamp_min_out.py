import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ClampMin_OutModule(torch.nn.Module):
    def forward(self, x, min, out):
        return torch.ops.aten.clamp_min.out(x, min, out=out)

mod = Torch_Ops_Aten_ClampMin_OutModule()

x = torch.randn(3)
min = 1
out = torch.empty(3)

args = (x, min, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
