import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeDropoutBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, mask, scale, out):
        return torch.ops.aten.native_dropout_backward.out(grad_output, mask, scale, out=out)

mod = Torch_Ops_Aten_NativeDropoutBackward_OutModule()

grad_output = torch.randn(3)
mask = torch.randn(3)
scale = 1.0
out = torch.empty(3)

args = (grad_output, mask, scale, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
