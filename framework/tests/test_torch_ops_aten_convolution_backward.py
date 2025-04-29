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
bias_sizes = torch.tensor(0)  # Fallback for unknown type SymInt[]?
stride = torch.tensor(0)  # Fallback for unknown type SymInt[]
padding = torch.tensor(0)  # Fallback for unknown type SymInt[]
dilation = torch.tensor(0)  # Fallback for unknown type SymInt[]
transposed = True
output_padding = torch.tensor(0)  # Fallback for unknown type SymInt[]
groups = torch.tensor(0)  # Fallback for unknown type SymInt
output_mask = torch.tensor(0)  # Fallback for unknown type bool[3]

args = (grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
