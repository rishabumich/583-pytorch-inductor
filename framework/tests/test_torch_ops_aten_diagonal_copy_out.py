import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_DiagonalCopy_OutModule(torch.nn.Module):
    def forward(self, x, offset, dim1, dim2, out):
        return torch.ops.aten.diagonal_copy.out(x, offset, dim1, dim2, out=out)

mod = Torch_Ops_Aten_DiagonalCopy_OutModule()

x = torch.randn(3)
offset = 3
dim1 = 3
dim2 = 3
out = torch.empty(3)

args = (x, offset, dim1, dim2, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
