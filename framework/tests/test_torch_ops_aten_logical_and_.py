import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogicalAndModule(torch.nn.Module):
    def forward(self, x, other):
        return torch.ops.aten.logical_and_(x, other)

mod = Torch_Ops_Aten_LogicalAndModule()

x = torch.randn(3)
other = torch.randn(3)

args = (x, other,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
