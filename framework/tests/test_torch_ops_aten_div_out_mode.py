import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Div_OutModeModule(torch.nn.Module):
    def forward(self, x, other, rounding_mode, out):
        return torch.ops.aten.div.out_mode(x, other, rounding_mode, out=out)

mod = Torch_Ops_Aten_Div_OutModeModule()

x = torch.randn(3)
other = torch.randn(3)
rounding_mode = None  # Fallback for unknown type str?
out = torch.empty(3)

args = (x, other, rounding_mode, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
