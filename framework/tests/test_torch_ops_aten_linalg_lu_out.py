import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLu_OutModule(torch.nn.Module):
    def forward(self, A, pivot, P, L, U):
        return torch.ops.aten.linalg_lu.out(A, pivot, P, L, U)

mod = Torch_Ops_Aten_LinalgLu_OutModule()

A = torch.randn(3)
pivot = True
P = torch.randn(3)
L = torch.randn(3)
U = torch.randn(3)

args = (A, pivot, P, L, U,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
