import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeGroupNorm_OutModule(torch.nn.Module):
    def forward(self, input, weight, bias, N, C, HxW, group, eps, out0, out1, out2):
        return torch.ops.aten.native_group_norm.out(input, weight, bias, N, C, HxW, group, eps, out0, out1, out2)

mod = Torch_Ops_Aten_NativeGroupNorm_OutModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
N = None  # Fallback for unknown type SymInt
C = None  # Fallback for unknown type SymInt
HxW = None  # Fallback for unknown type SymInt
group = 3
eps = 1.0
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (input, weight, bias, N, C, HxW, group, eps, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
