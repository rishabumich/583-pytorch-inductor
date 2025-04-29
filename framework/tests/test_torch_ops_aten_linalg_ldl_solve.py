import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLdlSolveModule(torch.nn.Module):
    def forward(self, LD, pivots, B, hermitian):
        return torch.ops.aten.linalg_ldl_solve(LD, pivots, B, hermitian)

mod = Torch_Ops_Aten_LinalgLdlSolveModule()

LD = torch.randn(3)
pivots = torch.randn(3)
B = torch.randn(3)
hermitian = True

args = (LD, pivots, B, hermitian,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
