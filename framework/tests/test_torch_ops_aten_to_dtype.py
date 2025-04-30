import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_DtypeModule(torch.nn.Module):
    def forward(self, x, dtype, non_blocking, copy, memory_format):
        return torch.ops.aten.to.dtype(x, dtype, non_blocking, copy, memory_format)

mod = Torch_Ops_Aten_To_DtypeModule()

x = torch.randn(3)
dtype = None  # Fallback for unknown type ScalarType
non_blocking = True
copy = True
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (x, dtype, non_blocking, copy, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
