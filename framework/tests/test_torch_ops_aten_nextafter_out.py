import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Nextafter_OutModule(torch.nn.Module):
    def forward(self, x, other, out):
        return torch.ops.aten.nextafter.out(x, other, out=out)

mod = Torch_Ops_Aten_Nextafter_OutModule()

x = torch.randn(3)
other = torch.randn(3)
out = torch.empty(3)

args = (x, other, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
