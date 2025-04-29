import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedSelectModule(torch.nn.Module):
    def forward(self, x, mask):
        return torch.ops.aten.masked_select(x, mask)

mod = Torch_Ops_Aten_MaskedSelectModule()

x = torch.randn(3)
mask = torch.randn(3)

args = (x, mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
