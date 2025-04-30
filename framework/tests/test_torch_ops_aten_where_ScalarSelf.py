import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Where_ScalarselfModule(torch.nn.Module):
    def forward(self, condition, x, other):
        return torch.ops.aten.where.ScalarSelf(condition, x, other)

mod = Torch_Ops_Aten_Where_ScalarselfModule()

condition = torch.randn(3)
x = 1
other = torch.randn(3)

args = (condition, x, other,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
