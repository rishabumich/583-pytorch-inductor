import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AvgPool2DModule(torch.nn.Module):
    def forward(self, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
        return torch.ops.aten.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

mod = Torch_Ops_Aten_AvgPool2DModule()

x = torch.randn(3)
kernel_size = 3
stride = 3
padding = 3
ceil_mode = True
count_include_pad = True
divisor_override = 3

args = (x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
