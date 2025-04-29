import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AvgPool3DBackward_GradInputModule(torch.nn.Module):
    def forward(self, grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input):
        return torch.ops.aten.avg_pool3d_backward.grad_input(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input)

mod = Torch_Ops_Aten_AvgPool3DBackward_GradInputModule()

grad_output = torch.randn(3)
x = torch.randn(3)
kernel_size = torch.tensor(0)  # Fallback for unknown type int[3]
stride = torch.tensor(0)  # Fallback for unknown type int[3]
padding = torch.tensor(0)  # Fallback for unknown type int[3]
ceil_mode = True
count_include_pad = True
divisor_override = torch.tensor(0)  # Fallback for unknown type int?
grad_input = torch.randn(3)

args = (grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
