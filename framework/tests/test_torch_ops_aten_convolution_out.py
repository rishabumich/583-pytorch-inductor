import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Convolution_OutModule(torch.nn.Module):
    def forward(self, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out):
        return torch.ops.aten.convolution.out(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out=out)

mod = Torch_Ops_Aten_Convolution_OutModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
stride = torch.sym_int(3)
padding = torch.sym_int(3)
dilation = torch.sym_int(3)
transposed = True
output_padding = torch.sym_int(3)
groups = None  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
