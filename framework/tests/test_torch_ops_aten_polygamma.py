import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PolygammaModule(torch.nn.Module):
    def forward(self, n, x):
        return torch.ops.aten.polygamma(n, x)

mod = Torch_Ops_Aten_PolygammaModule()

n = None  # Fallback for unknown type |int
x = torch.randn(3)

args = (n, x,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
