import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AddmvModule(torch.nn.Module):
    def forward(self, x, mat, vec, beta, alpha):
        return torch.ops.aten.addmv_(x, mat, vec, beta, alpha)

mod = Torch_Ops_Aten_AddmvModule()

x = torch.randn(3)
mat = torch.randn(3)
vec = torch.randn(3)
beta = torch.tensor(0)  # Fallback for unknown type Scalar
alpha = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, mat, vec, beta, alpha,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
