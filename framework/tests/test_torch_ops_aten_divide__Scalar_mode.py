import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Divide_ScalarModeModule(torch.nn.Module):
    def forward(self, x, other, rounding_mode):
        return torch.ops.aten.divide_.Scalar_mode(x, other, rounding_mode)

mod = Torch_Ops_Aten_Divide_ScalarModeModule()

x = torch.randn(3)
other = 1
rounding_mode = None  # Fallback for unknown type str?

args = (x, other, rounding_mode,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
