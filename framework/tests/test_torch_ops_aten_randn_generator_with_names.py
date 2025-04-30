import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randn_GeneratorWithNamesModule(torch.nn.Module):
    def forward(self, size, generator, names, dtype, layout, device, pin_memory):
        return torch.ops.aten.randn.generator_with_names(size, generator, names, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_Randn_GeneratorWithNamesModule()

size = torch.sym_int(3)
generator = None  # Fallback for unknown type Generator?
names = None  # Fallback for unknown type str[]?
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True

args = (size, generator, names, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
