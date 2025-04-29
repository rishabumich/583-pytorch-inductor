import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaxPool2DWithIndicesBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices):
        return torch.ops.aten.max_pool2d_with_indices_backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices)

mod = Torch_Ops_Aten_MaxPool2DWithIndicesBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
kernel_size = torch.tensor(0)  # Fallback for unknown type int[2]
stride = torch.tensor(0)  # Fallback for unknown type int[2]
padding = torch.tensor(0)  # Fallback for unknown type int[2]
dilation = torch.tensor(0)  # Fallback for unknown type int[2]
ceil_mode = True
indices = torch.randn(3)

args = (grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
