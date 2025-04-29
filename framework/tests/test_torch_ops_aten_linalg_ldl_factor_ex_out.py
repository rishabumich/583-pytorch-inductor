import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLdlFactorEx_OutModule(torch.nn.Module):
    def forward(self, x, hermitian, check_errors, LD, pivots, info):
        return torch.ops.aten.linalg_ldl_factor_ex.out(x, hermitian, check_errors, LD, pivots, info)

mod = Torch_Ops_Aten_LinalgLdlFactorEx_OutModule()

x = torch.randn(3)
hermitian = True
check_errors = True
LD = torch.randn(3)
pivots = torch.randn(3)
info = torch.randn(3)

args = (x, hermitian, check_errors, LD, pivots, info,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
