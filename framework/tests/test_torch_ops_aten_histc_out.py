import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Histc_OutModule(torch.nn.Module):
    def forward(self, x, bins, min, max, out):
        return torch.ops.aten.histc.out(x, bins, min, max, out=out)

mod = Torch_Ops_Aten_Histc_OutModule()

x = torch.randn(3)
bins = 3
min = 1
max = 1
out = torch.empty(3)

args = (x, bins, min, max, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
