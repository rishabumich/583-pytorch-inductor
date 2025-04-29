import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeBatchNormBackwardModule(torch.nn.Module):
    def forward(self, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
        return torch.ops.aten.native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask)

mod = Torch_Ops_Aten_NativeBatchNormBackwardModule()

grad_out = torch.randn(3)
input = torch.randn(3)
weight = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
save_mean = torch.randn(3)
save_invstd = torch.randn(3)
train = True
eps = 1.0
output_mask = torch.tensor(0)  # Fallback for unknown type bool[3]

args = (grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
