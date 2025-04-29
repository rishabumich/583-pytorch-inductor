import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MultinomialModule(torch.nn.Module):
    def forward(self, x, num_samples, replacement, generator):
        return torch.ops.aten.multinomial(x, num_samples, replacement, generator)

mod = Torch_Ops_Aten_MultinomialModule()

x = torch.randn(3)
num_samples = 3
replacement = True
generator = torch.tensor(0)  # Fallback for unknown type Generator?

args = (x, num_samples, replacement, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
