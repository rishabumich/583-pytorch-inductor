import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RandLikeModule(torch.nn.Module):
    def forward(self, x, dtype, layout, device, pin_memory, memory_format):
        return torch.ops.aten.rand_like(x, dtype, layout, device, pin_memory, memory_format)

mod = Torch_Ops_Aten_RandLikeModule()

x = torch.randn(3)
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (x, dtype, layout, device, pin_memory, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
