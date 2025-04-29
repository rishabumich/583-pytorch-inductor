import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleBicubic2DModule(torch.nn.Module):
    def forward(self, x, output_size, align_corners, scales_h, scales_w):
        return torch.ops.aten.upsample_bicubic2d(x, output_size, align_corners, scales_h, scales_w)

mod = Torch_Ops_Aten_UpsampleBicubic2DModule()

x = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt[2]
align_corners = True
scales_h = torch.tensor(0)  # Fallback for unknown type float?
scales_w = torch.tensor(0)  # Fallback for unknown type float?

args = (x, output_size, align_corners, scales_h, scales_w,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
