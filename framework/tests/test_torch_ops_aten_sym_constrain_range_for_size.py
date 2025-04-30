import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SymConstrainRangeForSizeModule(torch.nn.Module):
    def forward(self, size, min, max):
        return torch.ops.aten.sym_constrain_range_for_size(size, min, max)

mod = Torch_Ops_Aten_SymConstrainRangeForSizeModule()

size = None  # Fallback for unknown type |Scalar
min = 3
max = 3

args = (size, min, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
