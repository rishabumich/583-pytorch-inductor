import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogicalXorModule(torch.nn.Module):
    def forward(self, x, other):
        return torch.ops.aten.logical_xor(x, other)

mod = Torch_Ops_Aten_LogicalXorModule()

x = torch.randn(3)
other = torch.randn(3)

args = (x, other,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
