import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AdaptiveMaxPool2DBackward_GradInputModule(torch.nn.Module):
    def forward(self, grad_output, x, indices, grad_input):
        return torch.ops.aten.adaptive_max_pool2d_backward.grad_input(grad_output, x, indices, grad_input)

mod = Torch_Ops_Aten_AdaptiveMaxPool2DBackward_GradInputModule()

grad_output = torch.randn(3)
x = torch.randn(3)
indices = torch.randn(3)
grad_input = torch.randn(3)

args = (grad_output, x, indices, grad_input,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
