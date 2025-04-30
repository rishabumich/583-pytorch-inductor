import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Gather_DimnameModule(torch.nn.Module):
    def forward(self, x, dim, index, sparse_grad):
        return torch.ops.aten.gather.dimname(x, dim, index, sparse_grad)

mod = Torch_Ops_Aten_Gather_DimnameModule()

x = torch.randn(3)
dim = None  # Fallback for unknown type str
index = torch.randn(3)
sparse_grad = True

args = (x, dim, index, sparse_grad,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
