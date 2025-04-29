import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaxPool2DWithIndicesModule(torch.nn.Module):
    def forward(self, x, kernel_size, stride, padding, dilation, ceil_mode):
        return torch.ops.aten.max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode)

mod = Torch_Ops_Aten_MaxPool2DWithIndicesModule()

x = torch.randn(3)
kernel_size = torch.tensor(0)  # Fallback for unknown type int[2]
stride = torch.tensor(0)  # Fallback for unknown type int[2]
padding = torch.tensor(0)  # Fallback for unknown type int[2]
dilation = torch.tensor(0)  # Fallback for unknown type int[2]
ceil_mode = True

args = (x, kernel_size, stride, padding, dilation, ceil_mode,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
