import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MiopenBatchNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon):
        return torch.ops.aten.miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)

mod = Torch_Ops_Aten_MiopenBatchNormModule()

input = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
training = True
exponential_average_factor = 1.0
epsilon = 1.0

args = (input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
