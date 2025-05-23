import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ExpandModule(torch.nn.Module):
    def forward(self, x, size, implicit):
        return torch.ops.aten.expand(x, size, implicit)

mod = Torch_Ops_Aten_ExpandModule()

x = torch.randn(3)
size = torch.sym_int(3)
implicit = True

args = (x, size, implicit,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
