import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_GridSampler3DBackwardModule(torch.nn.Module):
    def forward(self, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask):
        return torch.ops.aten.grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask)

mod = Torch_Ops_Aten_GridSampler3DBackwardModule()

grad_output = torch.randn(3)
input = torch.randn(3)
grid = torch.randn(3)
interpolation_mode = 3
padding_mode = 3
align_corners = True
output_mask = torch.tensor(0)  # Fallback for unknown type bool[2]

args = (grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
