import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MiopenBatchNormBackwardModule(torch.nn.Module):
    def forward(self, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon):
        return torch.ops.aten.miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon)

mod = Torch_Ops_Aten_MiopenBatchNormBackwardModule()

input = torch.randn(3)
grad_output = torch.randn(3)
weight = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
save_mean = torch.randn(3)
save_var = torch.randn(3)
epsilon = 1.0

args = (input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
