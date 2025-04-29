import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TanhBackward_GradInputModule(torch.nn.Module):
    def forward(self, grad_output, output, grad_input):
        return torch.ops.aten.tanh_backward.grad_input(grad_output, output, grad_input)

mod = Torch_Ops_Aten_TanhBackward_GradInputModule()

grad_output = torch.randn(3)
output = torch.randn(3)
grad_input = torch.randn(3)

args = (grad_output, output, grad_input,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
