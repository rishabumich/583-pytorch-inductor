import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CauchyModule(torch.nn.Module):
    def forward(self, x, median, sigma, generator):
        return torch.ops.aten.cauchy_(x, median, sigma, generator)

mod = Torch_Ops_Aten_CauchyModule()

x = torch.randn(3)
median = 1.0
sigma = 1.0
generator = None  # Fallback for unknown type Generator?

args = (x, median, sigma, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
