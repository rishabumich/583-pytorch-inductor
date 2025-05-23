import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaskedFill_ScalarModule(torch.nn.Module):
    def forward(self, x, mask, value):
        return torch.ops.aten.masked_fill_.Scalar(x, mask, value)

mod = Torch_Ops_Aten_MaskedFill_ScalarModule()

x = torch.randn(3)
mask = torch.randn(3)
value = 1

args = (x, mask, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
