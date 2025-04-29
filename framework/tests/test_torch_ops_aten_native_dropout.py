import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeDropoutModule(torch.nn.Module):
    def forward(self, input, p, train):
        return torch.ops.aten.native_dropout(input, p, train)

mod = Torch_Ops_Aten_NativeDropoutModule()

input = torch.randn(3)
p = 1.0
train = torch.tensor(0)  # Fallback for unknown type bool?

args = (input, p, train,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
