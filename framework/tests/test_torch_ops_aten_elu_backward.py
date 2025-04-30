import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EluBackwardModule(torch.nn.Module):
    def forward(self, grad_output, alpha, scale, input_scale, is_result, self_or_result):
        return torch.ops.aten.elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result)

mod = Torch_Ops_Aten_EluBackwardModule()

grad_output = torch.randn(3)
alpha = 1
scale = 1
input_scale = 1
is_result = True
self_or_result = torch.randn(3)

args = (grad_output, alpha, scale, input_scale, is_result, self_or_result,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
