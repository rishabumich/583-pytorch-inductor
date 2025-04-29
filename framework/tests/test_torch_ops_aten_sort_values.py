import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sort_ValuesModule(torch.nn.Module):
    def forward(self, x, dim, descending, values, indices):
        return torch.ops.aten.sort.values(x, dim, descending, values, indices)

mod = Torch_Ops_Aten_Sort_ValuesModule()

x = torch.randn(3)
dim = 3
descending = True
values = torch.randn(3)
indices = torch.randn(3)

args = (x, dim, descending, values, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
