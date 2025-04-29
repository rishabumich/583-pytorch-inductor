import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgSolveTriangularModule(torch.nn.Module):
    def forward(self, x, B, upper, left, unitriangular):
        return torch.ops.aten.linalg_solve_triangular(x, B, upper, left, unitriangular)

mod = Torch_Ops_Aten_LinalgSolveTriangularModule()

x = torch.randn(3)
B = torch.randn(3)
upper = True
left = True
unitriangular = True

args = (x, B, upper, left, unitriangular,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
