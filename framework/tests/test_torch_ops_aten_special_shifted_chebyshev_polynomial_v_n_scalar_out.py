import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialV_NScalarOutModule(torch.nn.Module):
    def forward(self, x, n, out):
        return torch.ops.aten.special_shifted_chebyshev_polynomial_v.n_scalar_out(x, n, out=out)

mod = Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialV_NScalarOutModule()

x = torch.randn(3)
n = 1
out = torch.empty(3)

args = (x, n, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
