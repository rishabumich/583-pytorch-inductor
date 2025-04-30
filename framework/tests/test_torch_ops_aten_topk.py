import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TopkModule(torch.nn.Module):
    def forward(self, x, k, dim, largest, sorted):
        return torch.ops.aten.topk(x, k, dim, largest, sorted)

mod = Torch_Ops_Aten_TopkModule()

x = torch.randn(3)
k = None  # Fallback for unknown type SymInt
dim = 3
largest = True
sorted = True

args = (x, k, dim, largest, sorted,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
