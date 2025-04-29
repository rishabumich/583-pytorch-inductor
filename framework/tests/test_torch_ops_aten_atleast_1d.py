import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Atleast1DModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.atleast_1d(x)

mod = Torch_Ops_Aten_Atleast1DModule()

x = torch.randn(3, 3)

args = (x,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
