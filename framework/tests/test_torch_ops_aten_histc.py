import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HistcModule(torch.nn.Module):
    def forward(self, x, bins, min, max):
        return torch.ops.aten.histc(x, bins, min, max)

mod = Torch_Ops_Aten_HistcModule()

x = torch.randn(3)
bins = 3
min = torch.tensor(0)  # Fallback for unknown type Scalar
max = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, bins, min, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
