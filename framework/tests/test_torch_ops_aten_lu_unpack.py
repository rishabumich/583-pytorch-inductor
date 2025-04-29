import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LuUnpackModule(torch.nn.Module):
    def forward(self, LU_data, LU_pivots, unpack_data, unpack_pivots):
        return torch.ops.aten.lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots)

mod = Torch_Ops_Aten_LuUnpackModule()

LU_data = torch.randn(3)
LU_pivots = torch.randn(3)
unpack_data = True
unpack_pivots = True

args = (LU_data, LU_pivots, unpack_data, unpack_pivots,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
