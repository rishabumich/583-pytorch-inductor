import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeGroupNormBackwardModule(torch.nn.Module):
    def forward(self, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask):
        return torch.ops.aten.native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask)

mod = Torch_Ops_Aten_NativeGroupNormBackwardModule()

grad_out = torch.randn(3)
input = torch.randn(3)
mean = torch.randn(3)
rstd = torch.randn(3)
weight = torch.randn(3)
N = None  # Fallback for unknown type SymInt
C = None  # Fallback for unknown type SymInt
HxW = None  # Fallback for unknown type SymInt
group = 3
output_mask = True

args = (grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
