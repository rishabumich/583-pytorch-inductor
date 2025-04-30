import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_DtypeLayoutModule(torch.nn.Module):
    def forward(self, x, dtype, layout, device, pin_memory, non_blocking, copy, memory_format):
        return torch.ops.aten.to.dtype_layout(x, dtype, layout, device, pin_memory, non_blocking, copy, memory_format)

mod = Torch_Ops_Aten_To_DtypeLayoutModule()

x = torch.randn(3)
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True
non_blocking = True
copy = True
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (x, dtype, layout, device, pin_memory, non_blocking, copy, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
