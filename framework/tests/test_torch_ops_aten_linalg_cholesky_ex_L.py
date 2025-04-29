import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgCholeskyEx_LModule(torch.nn.Module):
    def forward(self, x, upper, check_errors, L, info):
        return torch.ops.aten.linalg_cholesky_ex.L(x, upper, check_errors, L, info)

mod = Torch_Ops_Aten_LinalgCholeskyEx_LModule()

x = torch.randn(3)
upper = True
check_errors = True
L = torch.randn(3)
info = torch.randn(3)

args = (x, upper, check_errors, L, info,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
