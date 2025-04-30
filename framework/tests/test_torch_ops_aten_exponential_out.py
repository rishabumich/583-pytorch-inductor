import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Exponential_OutModule(torch.nn.Module):
    def forward(self, x, lambd, generator, out):
        return torch.ops.aten.exponential.out(x, lambd, generator, out=out)

mod = Torch_Ops_Aten_Exponential_OutModule()

x = torch.randn(3)
lambd = 1.0
generator = None  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (x, lambd, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
