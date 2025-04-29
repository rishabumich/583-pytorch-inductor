import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FullLikeModule(torch.nn.Module):
    def forward(self, x, fill_value, dtype, layout, device, pin_memory, memory_format):
        return torch.ops.aten.full_like(x, fill_value, dtype, layout, device, pin_memory, memory_format)

mod = Torch_Ops_Aten_FullLikeModule()

x = torch.randn(3)
fill_value = torch.tensor(0)  # Fallback for unknown type Scalar
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType?
layout = torch.tensor(0)  # Fallback for unknown type Layout?
device = torch.tensor(0)  # Fallback for unknown type Device?
pin_memory = torch.tensor(0)  # Fallback for unknown type bool?
memory_format = torch.tensor(0)  # Fallback for unknown type MemoryFormat?

args = (x, fill_value, dtype, layout, device, pin_memory, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
