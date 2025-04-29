import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedScatter_OutModule(torch.nn.Module):
    def forward(self, x, mask, source, out):
        return torch.ops.aten.masked_scatter.out(x, mask, source, out=out)

mod = Torch_Ops_Aten_MaskedScatter_OutModule()

x = torch.randn(3)
mask = torch.randn(3)
source = torch.randn(3)
out = torch.empty(3)

args = (x, mask, source, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
