import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Cummax_OutModule(torch.nn.Module):
    def forward(self, x, dim, values, indices):
        return torch.ops.aten.cummax.out(x, dim, values, indices)

mod = Torch_Ops_Aten_Cummax_OutModule()

x = torch.randn(3)
dim = 3
values = torch.randn(3)
indices = torch.randn(3)

args = (x, dim, values, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
