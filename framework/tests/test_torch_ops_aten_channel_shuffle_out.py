import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ChannelShuffle_OutModule(torch.nn.Module):
    def forward(self, x, groups, out):
        return torch.ops.aten.channel_shuffle.out(x, groups, out=out)

mod = Torch_Ops_Aten_ChannelShuffle_OutModule()

x = torch.randn(3)
groups = None  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (x, groups, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
