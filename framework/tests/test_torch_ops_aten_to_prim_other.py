import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_PrimOtherModule(torch.nn.Module):
    def forward(self, x, non_blocking, copy):
        return torch.ops.aten.to.prim_other(x, non_blocking, copy)

mod = Torch_Ops_Aten_To_PrimOtherModule()

x = torch.randn(3)
non_blocking = True
copy = True

args = (x, non_blocking, copy,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
