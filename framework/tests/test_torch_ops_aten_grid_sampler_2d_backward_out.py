import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_GridSampler2DBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1):
        return torch.ops.aten.grid_sampler_2d_backward.out(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1)

mod = Torch_Ops_Aten_GridSampler2DBackward_OutModule()

grad_output = torch.randn(3)
input = torch.randn(3)
grid = torch.randn(3)
interpolation_mode = 3
padding_mode = 3
align_corners = True
output_mask = torch.tensor(0)  # Fallback for unknown type bool[2]
out0 = torch.randn(3)
out1 = torch.randn(3)

args = (grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
