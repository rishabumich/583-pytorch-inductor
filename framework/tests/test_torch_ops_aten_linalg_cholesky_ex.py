import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgCholeskyExModule(torch.nn.Module):
    def forward(self, x, upper, check_errors):
        return torch.ops.aten.linalg_cholesky_ex(x, upper, check_errors)

mod = Torch_Ops_Aten_LinalgCholeskyExModule()

x = torch.randn(3)
upper = True
check_errors = True

args = (x, upper, check_errors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
