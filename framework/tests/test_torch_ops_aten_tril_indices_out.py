import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TrilIndices_OutModule(torch.nn.Module):
    def forward(self, row, col, offset, out):
        return torch.ops.aten.tril_indices.out(row, col, offset, out=out)

mod = Torch_Ops_Aten_TrilIndices_OutModule()

row = None  # Fallback for unknown type |int
col = 3
offset = 3
out = torch.empty(3)

args = (row, col, offset, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
