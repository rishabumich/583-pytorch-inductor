import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleLinear1D_OutModule(torch.nn.Module):
    def forward(self, x, output_size, align_corners, scales, out):
        return torch.ops.aten.upsample_linear1d.out(x, output_size, align_corners, scales, out=out)

mod = Torch_Ops_Aten_UpsampleLinear1D_OutModule()

x = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt[1]
align_corners = True
scales = torch.tensor(0)  # Fallback for unknown type float?
out = torch.empty(3)

args = (x, output_size, align_corners, scales, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
