import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SplitWithSizesCopyModule(torch.nn.Module):
    def forward(self, x, split_sizes, dim):
        return torch.ops.aten.split_with_sizes_copy(x, split_sizes, dim)

mod = Torch_Ops_Aten_SplitWithSizesCopyModule()

x = torch.randn(3)
split_sizes = torch.tensor(0)  # Fallback for unknown type SymInt[]
dim = 3

args = (x, split_sizes, dim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
