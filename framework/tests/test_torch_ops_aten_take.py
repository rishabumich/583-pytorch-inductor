import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TakeModule(torch.nn.Module):
    def forward(self, x, index):
        return torch.ops.aten.take(x, index)

mod = Torch_Ops_Aten_TakeModule()

x = torch.randn(3)
index = torch.randn(3)

args = (x, index,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
