import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Multinomial_OutModule(torch.nn.Module):
    def forward(self, x, num_samples, replacement, generator, out):
        return torch.ops.aten.multinomial.out(x, num_samples, replacement, generator, out=out)

mod = Torch_Ops_Aten_Multinomial_OutModule()

x = torch.randn(3)
num_samples = 3
replacement = True
generator = None  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (x, num_samples, replacement, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
