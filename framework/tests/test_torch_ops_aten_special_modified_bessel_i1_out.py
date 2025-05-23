import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SpecialModifiedBesselI1_OutModule(torch.nn.Module):
    def forward(self, x, out):
        return torch.ops.aten.special_modified_bessel_i1.out(x, out=out)

mod = Torch_Ops_Aten_SpecialModifiedBesselI1_OutModule()

x = torch.randn(3)
out = torch.empty(3)

args = (x, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
