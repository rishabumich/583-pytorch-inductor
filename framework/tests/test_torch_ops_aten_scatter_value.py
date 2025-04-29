import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Scatter_ValueModule(torch.nn.Module):
    def forward(self, x, dim, index, value):
        return torch.ops.aten.scatter.value(x, dim, index, value)

mod = Torch_Ops_Aten_Scatter_ValueModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
value = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, dim, index, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
