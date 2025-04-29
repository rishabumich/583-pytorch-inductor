import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Narrow_TensorModule(torch.nn.Module):
    def forward(self, x, dim, start, length):
        return torch.ops.aten.narrow.Tensor(x, dim, start, length)

mod = Torch_Ops_Aten_Narrow_TensorModule()

x = torch.randn(3)
dim = 3
start = torch.randn(3)
length = torch.tensor(0)  # Fallback for unknown type SymInt

args = (x, dim, start, length,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
