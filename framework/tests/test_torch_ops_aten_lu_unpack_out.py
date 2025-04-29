import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LuUnpack_OutModule(torch.nn.Module):
    def forward(self, LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U):
        return torch.ops.aten.lu_unpack.out(LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U)

mod = Torch_Ops_Aten_LuUnpack_OutModule()

LU_data = torch.randn(3)
LU_pivots = torch.randn(3)
unpack_data = True
unpack_pivots = True
P = torch.randn(3)
L = torch.randn(3)
U = torch.randn(3)

args = (LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
