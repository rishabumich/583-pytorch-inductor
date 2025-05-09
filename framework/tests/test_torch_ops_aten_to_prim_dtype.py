import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_PrimDtypeModule(torch.nn.Module):
    def forward(self, x, dtype, non_blocking, copy):
        return torch.ops.aten.to.prim_dtype(x, dtype, non_blocking, copy)

mod = Torch_Ops_Aten_To_PrimDtypeModule()

x = torch.randn(3)
dtype = 3
non_blocking = True
copy = True

args = (x, dtype, non_blocking, copy,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
