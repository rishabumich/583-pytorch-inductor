import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgLdlFactorExModule(torch.nn.Module):
    def forward(self, x, hermitian, check_errors):
        return torch.ops.aten.linalg_ldl_factor_ex(x, hermitian, check_errors)

mod = Torch_Ops_Aten_LinalgLdlFactorExModule()

x = torch.randn(3)
hermitian = True
check_errors = True

args = (x, hermitian, check_errors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
