import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeBatchNorm_OutModule(torch.nn.Module):
    def forward(self, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd):
        return torch.ops.aten.native_batch_norm.out(input, weight, bias, running_mean, running_var, training, momentum, eps, out=out, save_mean, save_invstd)

mod = Torch_Ops_Aten_NativeBatchNorm_OutModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
training = True
momentum = 1.0
eps = 1.0
out = torch.empty(3)
save_mean = torch.randn(3)
save_invstd = torch.randn(3)

args = (input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
