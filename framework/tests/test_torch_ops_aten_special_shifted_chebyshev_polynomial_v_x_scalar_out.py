import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialV_XScalarOutModule(torch.nn.Module):
    def forward(self, x, n, out):
        return torch.ops.aten.special_shifted_chebyshev_polynomial_v.x_scalar_out(x, n, out=out)

mod = Torch_Ops_Aten_SpecialShiftedChebyshevPolynomialV_XScalarOutModule()

x = None  # Fallback for unknown type |Scalar
n = torch.randn(3)
out = torch.empty(3)

args = (x, n, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
