import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeBatchNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, running_mean, running_var, training, momentum, eps):
        return torch.ops.aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)

mod = Torch_Ops_Aten_NativeBatchNormModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
training = True
momentum = 1.0
eps = 1.0

args = (input, weight, bias, running_mean, running_var, training, momentum, eps,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
