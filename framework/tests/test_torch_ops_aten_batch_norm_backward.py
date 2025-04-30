import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BatchNormBackwardModule(torch.nn.Module):
    def forward(self, grad_out, input, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve):
        return torch.ops.aten.batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve)

mod = Torch_Ops_Aten_BatchNormBackwardModule()

grad_out = torch.randn(3)
input = torch.randn(3)
weight = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
save_mean = torch.randn(3)
save_var = torch.randn(3)
update = True
eps = 1.0
output_mask = True
reserve = torch.randn(3)

args = (grad_out, input, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
