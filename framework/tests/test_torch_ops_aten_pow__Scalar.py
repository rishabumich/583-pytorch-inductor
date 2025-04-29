import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Pow_ScalarModule(torch.nn.Module):
    def forward(self, x, exponent):
        return torch.ops.aten.pow_.Scalar(x, exponent)

mod = Torch_Ops_Aten_Pow_ScalarModule()

x = torch.randn(3)
exponent = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, exponent,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
