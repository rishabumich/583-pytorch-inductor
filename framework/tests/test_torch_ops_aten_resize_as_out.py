import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ResizeAs_OutModule(torch.nn.Module):
    def forward(self, x, the_template, memory_format, out):
        return torch.ops.aten.resize_as.out(x, the_template, memory_format, out=out)

mod = Torch_Ops_Aten_ResizeAs_OutModule()

x = torch.randn(3)
the_template = torch.randn(3)
memory_format = None  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (x, the_template, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
