import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FftHfftModule(torch.nn.Module):
    def forward(self, x, n, dim, norm):
        return torch.ops.aten.fft_hfft(x, n, dim, norm)

mod = Torch_Ops_Aten_FftHfftModule()

x = torch.randn(3)
n = torch.tensor(0)  # Fallback for unknown type SymInt?
dim = 3
norm = torch.tensor(0)  # Fallback for unknown type str?

args = (x, n, dim, norm,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
