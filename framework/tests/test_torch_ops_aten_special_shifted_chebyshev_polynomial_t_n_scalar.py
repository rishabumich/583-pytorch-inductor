import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialT_NScalarModule(torch.nn.Module):
    def forward(self, x, n):
        return torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar(x, n)

mod = Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialT_NScalarModule()

x = torch.randn(3)
n = 1

args = (x, n,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
