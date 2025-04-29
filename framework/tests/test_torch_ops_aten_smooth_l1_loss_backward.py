import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SmoothL1LossBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, target, reduction, beta):
        return torch.ops.aten.smooth_l1_loss_backward(grad_output, x, target, reduction, beta)

mod = Torch_Ops_Aten_SmoothL1LossBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
target = torch.randn(3)
reduction = 3
beta = 1.0

args = (grad_output, x, target, reduction, beta,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
