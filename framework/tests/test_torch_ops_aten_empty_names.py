import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Empty_NamesModule(torch.nn.Module):
    def forward(self, size, names, dtype, layout, device, pin_memory, memory_format):
        return torch.ops.aten.empty.names(size, names, dtype, layout, device, pin_memory, memory_format)

mod = Torch_Ops_Aten_Empty_NamesModule()

size = 3
names = None  # Fallback for unknown type str[]?
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (size, names, dtype, layout, device, pin_memory, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
