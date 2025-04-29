import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BinaryCrossEntropyBackward_GradInputModule(torch.nn.Module):
    def forward(self, grad_output, x, target, weight, reduction, grad_input):
        return torch.ops.aten.binary_cross_entropy_backward.grad_input(grad_output, x, target, weight, reduction, grad_input)

mod = Torch_Ops_Aten_BinaryCrossEntropyBackward_GradInputModule()

grad_output = torch.randn(3)
x = torch.randn(3)
target = torch.randn(3)
weight = torch.randn(3)
reduction = 3
grad_input = torch.randn(3)

args = (grad_output, x, target, weight, reduction, grad_input,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
