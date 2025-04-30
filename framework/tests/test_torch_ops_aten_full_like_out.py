import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FullLike_OutModule(torch.nn.Module):
    def forward(self, x, fill_value, memory_format, out):
        return torch.ops.aten.full_like.out(x, fill_value, memory_format, out=out)

mod = Torch_Ops_Aten_FullLike_OutModule()

x = torch.randn(3)
fill_value = 1
memory_format = None  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (x, fill_value, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
