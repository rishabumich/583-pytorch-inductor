import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Searchsorted_TensorModule(torch.nn.Module):
    def forward(self, sorted_sequence, x, out_int32, right, side, sorter):
        return torch.ops.aten.searchsorted.Tensor(sorted_sequence, x, out_int32, right, side, sorter)

mod = Torch_Ops_Aten_Searchsorted_TensorModule()

sorted_sequence = torch.randn(3)
x = torch.randn(3)
out_int32 = True
right = True
side = torch.tensor(0)  # Fallback for unknown type str?
sorter = torch.randn(3)

args = (sorted_sequence, x, out_int32, right, side, sorter,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
