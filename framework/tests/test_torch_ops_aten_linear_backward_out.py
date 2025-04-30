import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinearBackward_OutModule(torch.nn.Module):
    def forward(self, x, grad_output, weight, output_mask, out0, out1, out2):
        return torch.ops.aten.linear_backward.out(x, grad_output, weight, output_mask, out0, out1, out2)

mod = Torch_Ops_Aten_LinearBackward_OutModule()

x = torch.randn(3)
grad_output = torch.randn(3)
weight = torch.randn(3)
output_mask = True
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (x, grad_output, weight, output_mask, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
