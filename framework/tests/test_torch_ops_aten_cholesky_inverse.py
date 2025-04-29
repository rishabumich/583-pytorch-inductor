import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CholeskyInverseModule(torch.nn.Module):
    def forward(self, x, upper):
        return torch.ops.aten.cholesky_inverse(x, upper)

mod = Torch_Ops_Aten_CholeskyInverseModule()

x = torch.randn(3)
upper = True

args = (x, upper,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
