import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Div_ScalarModeOutModule(torch.nn.Module):
    def forward(self, x, other, rounding_mode, out):
        return torch.ops.aten.div.Scalar_mode_out(x, other, rounding_mode, out=out)

mod = Torch_Ops_Aten_Div_ScalarModeOutModule()

x = torch.randn(3)
other = torch.tensor(0)  # Fallback for unknown type Scalar
rounding_mode = torch.tensor(0)  # Fallback for unknown type str?
out = torch.empty(3)

args = (x, other, rounding_mode, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
