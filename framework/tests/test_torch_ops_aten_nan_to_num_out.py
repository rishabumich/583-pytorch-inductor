import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NanToNum_OutModule(torch.nn.Module):
    def forward(self, x, nan, posinf, neginf, out):
        return torch.ops.aten.nan_to_num.out(x, nan, posinf, neginf, out=out)

mod = Torch_Ops_Aten_NanToNum_OutModule()

x = torch.randn(3)
nan = 1.0
posinf = 1.0
neginf = 1.0
out = torch.empty(3)

args = (x, nan, posinf, neginf, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
