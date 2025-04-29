import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PreluModule(torch.nn.Module):
    def forward(self, x, weight):
        return torch.ops.aten.prelu(x, weight)

mod = Torch_Ops_Aten_PreluModule()

x = torch.randn(3)
weight = torch.randn(3)

args = (x, weight,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
