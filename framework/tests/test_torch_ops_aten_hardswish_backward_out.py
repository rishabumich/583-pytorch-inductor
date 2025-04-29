import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HardswishBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, x, out):
        return torch.ops.aten.hardswish_backward.out(grad_output, x, out=out)

mod = Torch_Ops_Aten_HardswishBackward_OutModule()

grad_output = torch.randn(3)
x = torch.randn(3)
out = torch.empty(3)

args = (grad_output, x, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
