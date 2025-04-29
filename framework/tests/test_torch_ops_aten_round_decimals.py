import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Round_DecimalsModule(torch.nn.Module):
    def forward(self, x, decimals):
        return torch.ops.aten.round.decimals(x, decimals)

mod = Torch_Ops_Aten_Round_DecimalsModule()

x = torch.randn(3)
decimals = 3

args = (x, decimals,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
