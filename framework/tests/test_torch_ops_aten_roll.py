import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RollModule(torch.nn.Module):
    def forward(self, x, shifts, dims):
        return torch.ops.aten.roll(x, shifts, dims)

mod = Torch_Ops_Aten_RollModule()

x = torch.randn(3)
shifts = torch.sym_int(3)
dims = 3

args = (x, shifts, dims,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
