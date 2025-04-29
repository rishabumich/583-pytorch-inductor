import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AminmaxModule(torch.nn.Module):
    def forward(self, x, dim, keepdim):
        return torch.ops.aten.aminmax(x, dim, keepdim)

mod = Torch_Ops_Aten_AminmaxModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int?
keepdim = True

args = (x, dim, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
