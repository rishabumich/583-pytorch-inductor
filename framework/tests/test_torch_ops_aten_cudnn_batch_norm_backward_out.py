import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CudnnBatchNormBackward_OutModule(torch.nn.Module):
    def forward(self, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace, out0, out1, out2):
        return torch.ops.aten.cudnn_batch_norm_backward.out(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace, out0, out1, out2)

mod = Torch_Ops_Aten_CudnnBatchNormBackward_OutModule()

input = torch.randn(3)
grad_output = torch.randn(3)
weight = torch.randn(3)
running_mean = torch.randn(3)
running_var = torch.randn(3)
save_mean = torch.randn(3)
save_var = torch.randn(3)
epsilon = 1.0
reserveSpace = torch.randn(3)
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
