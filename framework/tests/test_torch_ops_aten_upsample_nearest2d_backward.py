import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleNearest2DBackwardModule(torch.nn.Module):
    def forward(self, grad_output, output_size, input_size, scales_h, scales_w):
        return torch.ops.aten.upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w)

mod = Torch_Ops_Aten_UpsampleNearest2DBackwardModule()

grad_output = torch.randn(3)
output_size = torch.sym_int(3)
input_size = torch.sym_int(3)
scales_h = 1.0
scales_w = 1.0

args = (grad_output, output_size, input_size, scales_h, scales_w,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
