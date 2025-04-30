import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Full_NamesOutModule(torch.nn.Module):
    def forward(self, size, fill_value, names, out):
        return torch.ops.aten.full.names_out(size, fill_value, names, out=out)

mod = Torch_Ops_Aten_Full_NamesOutModule()

size = 3
fill_value = 1
names = None  # Fallback for unknown type str[]?
out = torch.empty(3)

args = (size, fill_value, names, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
