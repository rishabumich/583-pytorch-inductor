import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CopyModule(torch.nn.Module):
    def forward(self, x, src, non_blocking):
        return torch.ops.aten.copy_(x, src, non_blocking)

mod = Torch_Ops_Aten_CopyModule()

x = torch.randn(3)
src = torch.randn(3)
non_blocking = True

args = (x, src, non_blocking,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
