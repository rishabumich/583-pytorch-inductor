import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NanToNumModule(torch.nn.Module):
    def forward(self, x, nan, posinf, neginf):
        return torch.ops.aten.nan_to_num(x, nan, posinf, neginf)

mod = Torch_Ops_Aten_NanToNumModule()

x = torch.randn(3)
nan = 1.0
posinf = 1.0
neginf = 1.0

args = (x, nan, posinf, neginf,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
