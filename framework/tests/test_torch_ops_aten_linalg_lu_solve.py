import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLuSolveModule(torch.nn.Module):
    def forward(self, LU, pivots, B, left, adjoint):
        return torch.ops.aten.linalg_lu_solve(LU, pivots, B, left, adjoint)

mod = Torch_Ops_Aten_LinalgLuSolveModule()

LU = torch.randn(3)
pivots = torch.randn(3)
B = torch.randn(3)
left = True
adjoint = True

args = (LU, pivots, B, left, adjoint,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
