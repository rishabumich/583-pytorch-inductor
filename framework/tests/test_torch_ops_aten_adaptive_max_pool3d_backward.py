import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AdaptiveMaxPool3DBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, indices):
        return torch.ops.aten.adaptive_max_pool3d_backward(grad_output, x, indices)

mod = Torch_Ops_Aten_AdaptiveMaxPool3DBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
indices = torch.randn(3)

args = (grad_output, x, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
