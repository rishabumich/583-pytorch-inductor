import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_GridSampler2D_OutModule(torch.nn.Module):
    def forward(self, input, grid, interpolation_mode, padding_mode, align_corners, out):
        return torch.ops.aten.grid_sampler_2d.out(input, grid, interpolation_mode, padding_mode, align_corners, out=out)

mod = Torch_Ops_Aten_GridSampler2D_OutModule()

input = torch.randn(3)
grid = torch.randn(3)
interpolation_mode = 3
padding_mode = 3
align_corners = True
out = torch.empty(3)

args = (input, grid, interpolation_mode, padding_mode, align_corners, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
