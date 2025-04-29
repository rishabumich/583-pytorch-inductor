import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Rand_NamesModule(torch.nn.Module):
    def forward(self, size, names, dtype, layout, device, pin_memory):
        return torch.ops.aten.rand.names(size, names, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_Rand_NamesModule()

size = torch.tensor(0)  # Fallback for unknown type |SymInt[]
names = torch.tensor(0)  # Fallback for unknown type str[]?
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType?
layout = torch.tensor(0)  # Fallback for unknown type Layout?
device = torch.tensor(0)  # Fallback for unknown type Device?
pin_memory = torch.tensor(0)  # Fallback for unknown type bool?

args = (size, names, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
