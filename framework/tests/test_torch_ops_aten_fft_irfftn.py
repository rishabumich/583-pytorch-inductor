import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FftIrfftnModule(torch.nn.Module):
    def forward(self, x, s, dim, norm):
        return torch.ops.aten.fft_irfftn(x, s, dim, norm)

mod = Torch_Ops_Aten_FftIrfftnModule()

x = torch.randn(3)
s = torch.tensor(0)  # Fallback for unknown type SymInt[1]?
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
norm = torch.tensor(0)  # Fallback for unknown type str?

args = (x, s, dim, norm,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
