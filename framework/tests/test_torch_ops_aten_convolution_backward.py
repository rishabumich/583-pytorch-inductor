import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ConvolutionBackwardModule(torch.nn.Module):
    def forward(self, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
        return torch.ops.aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask)

mod = Torch_Ops_Aten_ConvolutionBackwardModule()

grad_output = torch.randn(3)
input = torch.randn(3)
weight = torch.randn(3)
bias_sizes = torch.sym_int(3)
stride = torch.sym_int(3)
padding = torch.sym_int(3)
dilation = torch.sym_int(3)
transposed = True
output_padding = torch.sym_int(3)
groups = None  # Fallback for unknown type SymInt
output_mask = True

args = (grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
