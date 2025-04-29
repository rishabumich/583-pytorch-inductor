import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedScatterModule(torch.nn.Module):
    def forward(self, x, mask, source):
        return torch.ops.aten.masked_scatter(x, mask, source)

mod = Torch_Ops_Aten_MaskedScatterModule()

x = torch.randn(3)
mask = torch.randn(3)
source = torch.randn(3)

args = (x, mask, source,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
