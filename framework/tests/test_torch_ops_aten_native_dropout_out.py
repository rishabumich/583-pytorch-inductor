import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeDropout_OutModule(torch.nn.Module):
    def forward(self, input, p, train, out0, out1):
        return torch.ops.aten.native_dropout.out(input, p, train, out0, out1)

mod = Torch_Ops_Aten_NativeDropout_OutModule()

input = torch.randn(3)
p = 1.0
train = torch.tensor(0)  # Fallback for unknown type bool?
out0 = torch.randn(3)
out1 = torch.randn(3)

args = (input, p, train, out0, out1,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
