import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LeakyReluBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, negative_slope, self_is_result):
        return torch.ops.aten.leaky_relu_backward(grad_output, x, negative_slope, self_is_result)

mod = Torch_Ops_Aten_LeakyReluBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
negative_slope = torch.tensor(0)  # Fallback for unknown type Scalar
self_is_result = True

args = (grad_output, x, negative_slope, self_is_result,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
