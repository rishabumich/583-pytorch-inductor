import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BernoulliModule(torch.nn.Module):
    def forward(self, x, generator):
        return torch.ops.aten.bernoulli(x, generator)

mod = Torch_Ops_Aten_BernoulliModule()

x = torch.randn(3)
generator = None  # Fallback for unknown type Generator?

args = (x, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
