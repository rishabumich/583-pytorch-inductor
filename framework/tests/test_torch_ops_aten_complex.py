import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ComplexModule(torch.nn.Module):
    def forward(self, real, imag):
        return torch.ops.aten.complex(real, imag)

mod = Torch_Ops_Aten_ComplexModule()

real = torch.randn(3)
imag = torch.randn(3)

args = (real, imag,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
