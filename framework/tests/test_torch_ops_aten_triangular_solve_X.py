import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TriangularSolve_XModule(torch.nn.Module):
    def forward(self, x, A, upper, transpose, unitriangular, X, M):
        return torch.ops.aten.triangular_solve.X(x, A, upper, transpose, unitriangular, X, M)

mod = Torch_Ops_Aten_TriangularSolve_XModule()

x = torch.randn(3)
A = torch.randn(3)
upper = True
transpose = True
unitriangular = True
X = torch.randn(3)
M = torch.randn(3)

args = (x, A, upper, transpose, unitriangular, X, M,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
