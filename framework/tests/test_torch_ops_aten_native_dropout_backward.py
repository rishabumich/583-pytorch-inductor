import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeDropoutBackwardModule(torch.nn.Module):
    def forward(self, grad_output, mask, scale):
        return torch.ops.aten.native_dropout_backward(grad_output, mask, scale)

mod = Torch_Ops_Aten_NativeDropoutBackwardModule()

grad_output = torch.randn(3)
mask = torch.randn(3)
scale = 1.0

args = (grad_output, mask, scale,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
