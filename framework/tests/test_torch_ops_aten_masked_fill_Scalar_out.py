import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedFill_ScalarOutModule(torch.nn.Module):
    def forward(self, x, mask, value, out):
        return torch.ops.aten.masked_fill.Scalar_out(x, mask, value, out=out)

mod = Torch_Ops_Aten_MaskedFill_ScalarOutModule()

x = torch.randn(3)
mask = torch.randn(3)
value = 1
out = torch.empty(3)

args = (x, mask, value, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
