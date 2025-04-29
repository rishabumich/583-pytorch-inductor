import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_DiagonalModule(torch.nn.Module):
    def forward(self, x, offset, dim1, dim2):
        return torch.ops.aten.diagonal(x, offset, dim1, dim2)

mod = Torch_Ops_Aten_DiagonalModule()

x = torch.randn(3)
offset = 3
dim1 = 3
dim2 = 3

args = (x, offset, dim1, dim2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
