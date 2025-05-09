import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Im2ColModule(torch.nn.Module):
    def forward(self, x, kernel_size, dilation, padding, stride):
        return torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)

mod = Torch_Ops_Aten_Im2ColModule()

x = torch.randn(3)
kernel_size = 3
dilation = 3
padding = 3
stride = 3

args = (x, kernel_size, dilation, padding, stride,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
