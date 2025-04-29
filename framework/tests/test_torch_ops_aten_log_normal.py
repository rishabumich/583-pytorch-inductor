import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogNormalModule(torch.nn.Module):
    def forward(self, x, mean, std, generator):
        return torch.ops.aten.log_normal(x, mean, std, generator)

mod = Torch_Ops_Aten_LogNormalModule()

x = torch.randn(3)
mean = 1.0
std = 1.0
generator = torch.tensor(0)  # Fallback for unknown type Generator?

args = (x, mean, std, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
