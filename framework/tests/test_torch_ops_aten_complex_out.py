import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Complex_OutModule(torch.nn.Module):
    def forward(self, real, imag, out):
        return torch.ops.aten.complex.out(real, imag, out=out)

mod = Torch_Ops_Aten_Complex_OutModule()

real = torch.randn(3)
imag = torch.randn(3)
out = torch.empty(3)

args = (real, imag, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
