import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ClampMaxModule(torch.nn.Module):
    def forward(self, x, max):
        return torch.ops.aten.clamp_max_(x, max)

mod = Torch_Ops_Aten_ClampMaxModule()

x = torch.randn(3)
max = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
