import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ClipModule(torch.nn.Module):
    def forward(self, x, min, max):
        return torch.ops.aten.clip(x, min, max)

mod = Torch_Ops_Aten_ClipModule()

x = torch.randn(3)
min = torch.tensor(0)  # Fallback for unknown type Scalar?
max = torch.tensor(0)  # Fallback for unknown type Scalar?

args = (x, min, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
