import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ChannelShuffleModule(torch.nn.Module):
    def forward(self, x, groups):
        return torch.ops.aten.channel_shuffle(x, groups)

mod = Torch_Ops_Aten_ChannelShuffleModule()

x = torch.randn(3)
groups = None  # Fallback for unknown type SymInt

args = (x, groups,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
