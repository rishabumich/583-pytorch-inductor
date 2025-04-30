import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Topk_ValuesModule(torch.nn.Module):
    def forward(self, x, k, dim, largest, sorted, values, indices):
        return torch.ops.aten.topk.values(x, k, dim, largest, sorted, values, indices)

mod = Torch_Ops_Aten_Topk_ValuesModule()

x = torch.randn(3)
k = None  # Fallback for unknown type SymInt
dim = 3
largest = True
sorted = True
values = torch.randn(3)
indices = torch.randn(3)

args = (x, k, dim, largest, sorted, values, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
