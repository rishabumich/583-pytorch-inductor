import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HeavisideModule(torch.nn.Module):
    def forward(self, x, values):
        return torch.ops.aten.heaviside(x, values)

mod = Torch_Ops_Aten_HeavisideModule()

x = torch.randn(3)
values = torch.randn(3)

args = (x, values,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
