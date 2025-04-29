import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeBatchNormBackward_OutModule(torch.nn.Module):
    def forward(self, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, out0, out1, out2):
        return torch.ops.aten.native_batch_norm_backward.out(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, out0, out1, out2)

mod = Torch_Ops_Aten_NativeBatchNormBackward_OutModule()

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
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
