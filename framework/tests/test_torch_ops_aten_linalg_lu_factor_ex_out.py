import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLuFactorEx_OutModule(torch.nn.Module):
    def forward(self, A, pivot, check_errors, LU, pivots, info):
        return torch.ops.aten.linalg_lu_factor_ex.out(A, pivot, check_errors, LU, pivots, info)

mod = Torch_Ops_Aten_LinalgLuFactorEx_OutModule()

A = torch.randn(3)
pivot = True
check_errors = True
LU = torch.randn(3)
pivots = torch.randn(3)
info = torch.randn(3)

args = (A, pivot, check_errors, LU, pivots, info,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
