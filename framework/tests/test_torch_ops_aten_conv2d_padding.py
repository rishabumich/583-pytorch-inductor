import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Conv2D_PaddingModule(torch.nn.Module):
    def forward(self, input, weight, bias, stride, padding, dilation, groups):
        return torch.ops.aten.conv2d.padding(input, weight, bias, stride, padding, dilation, groups)

mod = Torch_Ops_Aten_Conv2D_PaddingModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
stride = torch.sym_int(3)
padding = None  # Fallback for unknown type str
dilation = torch.sym_int(3)
groups = None  # Fallback for unknown type SymInt

args = (input, weight, bias, stride, padding, dilation, groups,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
