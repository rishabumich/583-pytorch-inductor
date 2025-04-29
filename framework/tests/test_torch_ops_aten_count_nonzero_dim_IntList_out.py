import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CountNonzero_DimIntlistOutModule(torch.nn.Module):
    def forward(self, x, dim, out):
        return torch.ops.aten.count_nonzero.dim_IntList_out(x, dim, out=out)

mod = Torch_Ops_Aten_CountNonzero_DimIntlistOutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[]
out = torch.empty(3)

args = (x, dim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
