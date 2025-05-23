import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Pow_ScalarModule(torch.nn.Module):
    def forward(self, x, exponent):
        return torch.ops.aten.pow.Scalar(x, exponent)

mod = Torch_Ops_Aten_Pow_ScalarModule()

x = None  # Fallback for unknown type |Scalar
exponent = torch.randn(3)

args = (x, exponent,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
