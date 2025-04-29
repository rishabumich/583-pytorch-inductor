import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLdlSolve_OutModule(torch.nn.Module):
    def forward(self, LD, pivots, B, hermitian, out):
        return torch.ops.aten.linalg_ldl_solve.out(LD, pivots, B, hermitian, out=out)

mod = Torch_Ops_Aten_LinalgLdlSolve_OutModule()

LD = torch.randn(3)
pivots = torch.randn(3)
B = torch.randn(3)
hermitian = True
out = torch.empty(3)

args = (LD, pivots, B, hermitian, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
