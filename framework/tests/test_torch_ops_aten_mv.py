import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MvModule(torch.nn.Module):
    def forward(self, x, vec):
        return torch.ops.aten.mv(x, vec)

mod = Torch_Ops_Aten_MvModule()

x = torch.randn(3)
vec = torch.randn(3)

args = (x, vec,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
