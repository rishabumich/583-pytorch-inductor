import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaxPool3DWithIndices_OutModule(torch.nn.Module):
    def forward(self, x, kernel_size, stride, padding, dilation, ceil_mode, out, indices):
        return torch.ops.aten.max_pool3d_with_indices.out(x, kernel_size, stride, padding, dilation, ceil_mode, out=out, indices)

mod = Torch_Ops_Aten_MaxPool3DWithIndices_OutModule()

x = torch.randn(3)
kernel_size = 3
stride = 3
padding = 3
dilation = 3
ceil_mode = True
out = torch.empty(3)
indices = torch.randn(3)

args = (x, kernel_size, stride, padding, dilation, ceil_mode, out, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
