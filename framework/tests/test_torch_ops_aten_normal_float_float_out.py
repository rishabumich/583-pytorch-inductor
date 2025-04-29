import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Normal_FloatFloatOutModule(torch.nn.Module):
    def forward(self, mean, std, size, generator, out):
        return torch.ops.aten.normal.float_float_out(mean, std, size, generator, out=out)

mod = Torch_Ops_Aten_Normal_FloatFloatOutModule()

mean = torch.tensor(0)  # Fallback for unknown type |float
std = 1.0
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
generator = torch.tensor(0)  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (mean, std, size, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
