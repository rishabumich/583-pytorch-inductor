import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgInvExModule(torch.nn.Module):
    def forward(self, A, check_errors):
        return torch.ops.aten.linalg_inv_ex(A, check_errors)

mod = Torch_Ops_Aten_LinalgInvExModule()

A = torch.randn(3)
check_errors = True

args = (A, check_errors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
