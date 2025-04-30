import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FlipModule(torch.nn.Module):
    def forward(self, x, dims):
        return torch.ops.aten.flip(x, dims)

mod = Torch_Ops_Aten_FlipModule()

x = torch.randn(3)
dims = 3

args = (x, dims,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
