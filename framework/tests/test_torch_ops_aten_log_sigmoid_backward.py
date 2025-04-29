import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogSigmoidBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, buffer):
        return torch.ops.aten.log_sigmoid_backward(grad_output, x, buffer)

mod = Torch_Ops_Aten_LogSigmoidBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
buffer = torch.randn(3)

args = (grad_output, x, buffer,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
