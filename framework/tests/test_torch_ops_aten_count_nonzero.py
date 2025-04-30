import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CountNonzeroModule(torch.nn.Module):
    def forward(self, x, dim):
        return torch.ops.aten.count_nonzero(x, dim)

mod = Torch_Ops_Aten_CountNonzeroModule()

x = torch.randn(3)
dim = 3

args = (x, dim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
