import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EluBackward_GradInputModule(torch.nn.Module):
    def forward(self, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input):
        return torch.ops.aten.elu_backward.grad_input(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input)

mod = Torch_Ops_Aten_EluBackward_GradInputModule()

grad_output = torch.randn(3)
alpha = torch.tensor(0)  # Fallback for unknown type Scalar
scale = torch.tensor(0)  # Fallback for unknown type Scalar
input_scale = torch.tensor(0)  # Fallback for unknown type Scalar
is_result = True
self_or_result = torch.randn(3)
grad_input = torch.randn(3)

args = (grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
