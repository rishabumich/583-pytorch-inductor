import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Uniform_OutModule(torch.nn.Module):
    def forward(self, x, from, to, generator, out):
        return torch.ops.aten.uniform.out(x, from, to, generator, out=out)

mod = Torch_Ops_Aten_Uniform_OutModule()

x = torch.randn(3)
from = 1.0
to = 1.0
generator = torch.tensor(0)  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (x, from, to, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
