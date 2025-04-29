import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NllLoss2DBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, target, weight, reduction, ignore_index, total_weight):
        return torch.ops.aten.nll_loss2d_backward(grad_output, x, target, weight, reduction, ignore_index, total_weight)

mod = Torch_Ops_Aten_NllLoss2DBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
reduction = 3
ignore_index = torch.tensor(0)  # Fallback for unknown type SymInt
total_weight = torch.randn(3)

args = (grad_output, x, target, weight, reduction, ignore_index, total_weight,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
