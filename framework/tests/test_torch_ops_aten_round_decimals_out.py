import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Round_DecimalsOutModule(torch.nn.Module):
    def forward(self, x, decimals, out):
        return torch.ops.aten.round.decimals_out(x, decimals, out=out)

mod = Torch_Ops_Aten_Round_DecimalsOutModule()

x = torch.randn(3)
decimals = 3
out = torch.empty(3)

args = (x, decimals, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
