import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Empty_NamesOutModule(torch.nn.Module):
    def forward(self, size, names, memory_format, out):
        return torch.ops.aten.empty.names_out(size, names, memory_format, out=out)

mod = Torch_Ops_Aten_Empty_NamesOutModule()

size = 3
names = None  # Fallback for unknown type str[]?
memory_format = None  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (size, names, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
