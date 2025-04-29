import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Pow_ScalarOutModule(torch.nn.Module):
    def forward(self, x, exponent, out):
        return torch.ops.aten.pow.Scalar_out(x, exponent, out=out)

mod = Torch_Ops_Aten_Pow_ScalarOutModule()

x = torch.tensor(0)  # Fallback for unknown type |Scalar
exponent = torch.randn(3)
out = torch.empty(3)

args = (x, exponent, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
