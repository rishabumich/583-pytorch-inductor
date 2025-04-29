import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLuModule(torch.nn.Module):
    def forward(self, A, pivot):
        return torch.ops.aten.linalg_lu(A, pivot)

mod = Torch_Ops_Aten_LinalgLuModule()

A = torch.randn(3)
pivot = True

args = (A, pivot,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
