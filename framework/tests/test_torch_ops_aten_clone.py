import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CloneModule(torch.nn.Module):
    def forward(self, x, memory_format):
        return torch.ops.aten.clone(x, memory_format)

mod = Torch_Ops_Aten_CloneModule()

x = torch.randn(3)
memory_format = None  # Fallback for unknown type MemoryFormat?

args = (x, memory_format,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
