import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NewEmptyStridedModule(torch.nn.Module):
    def forward(self, x, size, stride, dtype, layout, device, pin_memory):
        return torch.ops.aten.new_empty_strided(x, size, stride, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_NewEmptyStridedModule()

x = torch.randn(3)
size = torch.sym_int(3)
stride = torch.sym_int(3)
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True

args = (x, size, stride, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
