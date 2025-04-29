import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Polar_OutModule(torch.nn.Module):
    def forward(self, abs, angle, out):
        return torch.ops.aten.polar.out(abs, angle, out=out)

mod = Torch_Ops_Aten_Polar_OutModule()

abs = torch.randn(3)
angle = torch.randn(3)
out = torch.empty(3)

args = (abs, angle, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
