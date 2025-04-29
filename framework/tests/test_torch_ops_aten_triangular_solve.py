import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TriangularSolveModule(torch.nn.Module):
    def forward(self, x, A, upper, transpose, unitriangular):
        return torch.ops.aten.triangular_solve(x, A, upper, transpose, unitriangular)

mod = Torch_Ops_Aten_TriangularSolveModule()

x = torch.randn(3)
A = torch.randn(3)
upper = True
transpose = True
unitriangular = True

args = (x, A, upper, transpose, unitriangular,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
