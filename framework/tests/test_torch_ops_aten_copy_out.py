import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Copy_OutModule(torch.nn.Module):
    def forward(self, x, src, non_blocking, out):
        return torch.ops.aten.copy.out(x, src, non_blocking, out=out)

mod = Torch_Ops_Aten_Copy_OutModule()

x = torch.randn(3)
src = torch.randn(3)
non_blocking = True
out = torch.empty(3)

args = (x, src, non_blocking, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
