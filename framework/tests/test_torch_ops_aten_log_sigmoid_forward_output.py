import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogSigmoidForward_OutputModule(torch.nn.Module):
    def forward(self, x, output, buffer):
        return torch.ops.aten.log_sigmoid_forward.output(x, output, buffer)

mod = Torch_Ops_Aten_LogSigmoidForward_OutputModule()

x = torch.randn(3)
output = torch.randn(3)
buffer = torch.randn(3)

args = (x, output, buffer,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
