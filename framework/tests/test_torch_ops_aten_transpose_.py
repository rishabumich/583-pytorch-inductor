import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TransposeModule(torch.nn.Module):
    def forward(self, x, dim0, dim1):
        return torch.ops.aten.transpose_(x, dim0, dim1)

mod = Torch_Ops_Aten_TransposeModule()

x = torch.randn(3)
dim0 = 3
dim1 = 3

args = (x, dim0, dim1,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
