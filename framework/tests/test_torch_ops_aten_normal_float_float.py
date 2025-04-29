import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Normal_FloatFloatModule(torch.nn.Module):
    def forward(self, mean, std, size, generator, dtype, layout, device, pin_memory):
        return torch.ops.aten.normal.float_float(mean, std, size, generator, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_Normal_FloatFloatModule()

mean = torch.tensor(0)  # Fallback for unknown type |float
std = 1.0
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
generator = torch.tensor(0)  # Fallback for unknown type Generator?
dtype = torch.tensor(0)  # Fallback for unknown type ScalarType?
layout = torch.tensor(0)  # Fallback for unknown type Layout?
device = torch.tensor(0)  # Fallback for unknown type Device?
pin_memory = torch.tensor(0)  # Fallback for unknown type bool?

args = (mean, std, size, generator, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
