import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Aminmax_OutModule(torch.nn.Module):
    def forward(self, x, dim, keepdim, min, max):
        return torch.ops.aten.aminmax.out(x, dim, keepdim, min, max)

mod = Torch_Ops_Aten_Aminmax_OutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int?
keepdim = True
min = torch.randn(3)
max = torch.randn(3)

args = (x, dim, keepdim, min, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
