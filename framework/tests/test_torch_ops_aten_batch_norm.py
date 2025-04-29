import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BatchNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
        return torch.ops.aten.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)

mod = Torch_Ops_Aten_BatchNormModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
training = True
momentum = 1.0
eps = 1.0
cudnn_enabled = True

args = (input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
