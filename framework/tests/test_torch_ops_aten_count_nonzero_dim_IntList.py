import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CountNonzero_DimIntlistModule(torch.nn.Module):
    def forward(self, x, dim):
        return torch.ops.aten.count_nonzero.dim_IntList(x, dim)

mod = Torch_Ops_Aten_CountNonzero_DimIntlistModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[]

args = (x, dim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
