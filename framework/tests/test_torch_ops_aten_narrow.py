import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NarrowModule(torch.nn.Module):
    def forward(self, x, dim, start, length):
        return torch.ops.aten.narrow(x, dim, start, length)

mod = Torch_Ops_Aten_NarrowModule()

x = torch.randn(3)
dim = 3
start = None  # Fallback for unknown type SymInt
length = None  # Fallback for unknown type SymInt

args = (x, dim, start, length,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
