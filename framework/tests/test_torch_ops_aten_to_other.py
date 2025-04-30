import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_OtherModule(torch.nn.Module):
    def forward(self, x, other, non_blocking, copy, memory_format):
        return torch.ops.aten.to.other(x, other, non_blocking, copy, memory_format)

mod = Torch_Ops_Aten_To_OtherModule()

x = torch.randn(3)
other = torch.randn(3)
non_blocking = True
copy = True
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (x, other, non_blocking, copy, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
