import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FftRfftModule(torch.nn.Module):
    def forward(self, x, n, dim, norm):
        return torch.ops.aten.fft_rfft(x, n, dim, norm)

mod = Torch_Ops_Aten_FftRfftModule()

x = torch.randn(3)
n = torch.sym_int(3)
dim = 3
norm = None  # Fallback for unknown type str?

args = (x, n, dim, norm,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
