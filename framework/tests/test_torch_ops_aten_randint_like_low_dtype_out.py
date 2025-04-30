import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RandintLike_LowDtypeOutModule(torch.nn.Module):
    def forward(self, x, low, high, memory_format, out):
        return torch.ops.aten.randint_like.low_dtype_out(x, low, high, memory_format, out=out)

mod = Torch_Ops_Aten_RandintLike_LowDtypeOutModule()

x = torch.randn(3)
low = None  # Fallback for unknown type SymInt
high = None  # Fallback for unknown type SymInt
memory_format = None  # Fallback for unknown type MemoryFormat?
out = torch.empty(3)

args = (x, low, high, memory_format, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
