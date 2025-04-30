import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Split_StrModule(torch.nn.Module):
    def forward(self, x, separator, max):
        return torch.ops.aten.split.str(x, separator, max)

mod = Torch_Ops_Aten_Split_StrModule()

x = None  # Fallback for unknown type |str
separator = None  # Fallback for unknown type str?
max = 3

args = (x, separator, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
