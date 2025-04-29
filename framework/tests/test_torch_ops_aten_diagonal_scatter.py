import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_DiagonalScatterModule(torch.nn.Module):
    def forward(self, x, src, offset, dim1, dim2):
        return torch.ops.aten.diagonal_scatter(x, src, offset, dim1, dim2)

mod = Torch_Ops_Aten_DiagonalScatterModule()

x = torch.randn(3)
src = torch.randn(3)
offset = 3
dim1 = 3
dim2 = 3

args = (x, src, offset, dim1, dim2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
