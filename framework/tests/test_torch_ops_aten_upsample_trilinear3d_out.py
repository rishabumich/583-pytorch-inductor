import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleTrilinear3D_OutModule(torch.nn.Module):
    def forward(self, x, output_size, align_corners, scales_d, scales_h, scales_w, out):
        return torch.ops.aten.upsample_trilinear3d.out(x, output_size, align_corners, scales_d, scales_h, scales_w, out=out)

mod = Torch_Ops_Aten_UpsampleTrilinear3D_OutModule()

x = torch.randn(3)
output_size = torch.sym_int(3)
align_corners = True
scales_d = 1.0
scales_h = 1.0
scales_w = 1.0
out = torch.empty(3)

args = (x, output_size, align_corners, scales_d, scales_h, scales_w, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
