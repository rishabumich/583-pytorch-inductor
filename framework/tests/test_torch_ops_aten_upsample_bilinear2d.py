import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleBilinear2DModule(torch.nn.Module):
    def forward(self, x, output_size, align_corners, scales_h, scales_w):
        return torch.ops.aten.upsample_bilinear2d(x, output_size, align_corners, scales_h, scales_w)

mod = Torch_Ops_Aten_UpsampleBilinear2DModule()

x = torch.randn(3)
output_size = torch.sym_int(3)
align_corners = True
scales_h = 1.0
scales_w = 1.0

args = (x, output_size, align_corners, scales_h, scales_w,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
