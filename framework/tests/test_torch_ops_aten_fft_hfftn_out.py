import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FftHfftn_OutModule(torch.nn.Module):
    def forward(self, x, s, dim, norm, out):
        return torch.ops.aten.fft_hfftn.out(x, s, dim, norm, out=out)

mod = Torch_Ops_Aten_FftHfftn_OutModule()

x = torch.randn(3)
s = torch.sym_int(3)
dim = 3
norm = None  # Fallback for unknown type str?
out = torch.empty(3)

args = (x, s, dim, norm, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
