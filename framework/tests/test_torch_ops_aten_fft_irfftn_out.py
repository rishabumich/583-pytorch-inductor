import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FftIrfftn_OutModule(torch.nn.Module):
    def forward(self, x, s, dim, norm, out):
        return torch.ops.aten.fft_irfftn.out(x, s, dim, norm, out=out)

mod = Torch_Ops_Aten_FftIrfftn_OutModule()

x = torch.randn(3)
s = torch.tensor(0)  # Fallback for unknown type SymInt[1]?
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
norm = torch.tensor(0)  # Fallback for unknown type str?
out = torch.empty(3)

args = (x, s, dim, norm, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
