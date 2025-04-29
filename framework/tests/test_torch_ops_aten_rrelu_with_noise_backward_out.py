import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RreluWithNoiseBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, x, noise, lower, upper, training, self_is_result, out):
        return torch.ops.aten.rrelu_with_noise_backward.out(grad_output, x, noise, lower, upper, training, self_is_result, out=out)

mod = Torch_Ops_Aten_RreluWithNoiseBackward_OutModule()

grad_output = torch.randn(3)
x = torch.randn(3)
noise = torch.randn(3)
lower = torch.tensor(0)  # Fallback for unknown type Scalar
upper = torch.tensor(0)  # Fallback for unknown type Scalar
training = True
self_is_result = True
out = torch.empty(3)

args = (grad_output, x, noise, lower, upper, training, self_is_result, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
