import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialTModule(torch.nn.Module):
    def forward(self, x, n):
        return torch.ops.aten.special_shifted_chebyshev_polynomial_t(x, n)

mod = Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialTModule()

x = torch.randn(3)
n = torch.randn(3)

args = (x, n,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
