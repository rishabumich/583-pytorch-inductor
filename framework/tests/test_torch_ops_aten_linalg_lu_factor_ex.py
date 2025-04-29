import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLuFactorExModule(torch.nn.Module):
    def forward(self, A, pivot, check_errors):
        return torch.ops.aten.linalg_lu_factor_ex(A, pivot, check_errors)

mod = Torch_Ops_Aten_LinalgLuFactorExModule()

A = torch.randn(3)
pivot = True
check_errors = True

args = (A, pivot, check_errors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
