import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Resize_OutModule(torch.nn.Module):
    def forward(self, x, size, memory_format, out):
        return torch.ops.aten.resize.out(x, size, memory_format, out=out)

mod = Torch_Ops_Aten_Resize_OutModule()

x = torch.randn(3)
size = torch.sym_int(3)
memory_format = None  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (x, size, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
