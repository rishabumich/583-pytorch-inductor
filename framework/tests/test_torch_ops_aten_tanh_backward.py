import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TanhBackwardModule(torch.nn.Module):
    def forward(self, grad_output, output):
        return torch.ops.aten.tanh_backward(grad_output, output)

mod = Torch_Ops_Aten_TanhBackwardModule()

grad_output = torch.randn(3)
output = torch.randn(3)

args = (grad_output, output,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
