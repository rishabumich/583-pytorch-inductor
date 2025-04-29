import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ExponentialModule(torch.nn.Module):
    def forward(self, x, lambd, generator):
        return torch.ops.aten.exponential_(x, lambd, generator)

mod = Torch_Ops_Aten_ExponentialModule()

x = torch.randn(3)
lambd = 1.0
generator = torch.tensor(0)  # Fallback for unknown type Generator?

args = (x, lambd, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
