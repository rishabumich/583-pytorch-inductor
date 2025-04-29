import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgInvEx_InverseModule(torch.nn.Module):
    def forward(self, A, check_errors, inverse, info):
        return torch.ops.aten.linalg_inv_ex.inverse(A, check_errors, inverse, info)

mod = Torch_Ops_Aten_LinalgInvEx_InverseModule()

A = torch.randn(3)
check_errors = True
inverse = torch.randn(3)
info = torch.randn(3)

args = (A, check_errors, inverse, info,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
