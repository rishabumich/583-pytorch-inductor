import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UnsafeSplitWithSizes_OutModule(torch.nn.Module):
    def forward(self, x, split_sizes, dim, out):
        return torch.ops.aten.unsafe_split_with_sizes.out(x, split_sizes, dim, out=out)

mod = Torch_Ops_Aten_UnsafeSplitWithSizes_OutModule()

x = torch.randn(3)
split_sizes = torch.sym_int(3)
dim = 3
out = torch.empty(3)

args = (x, split_sizes, dim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
