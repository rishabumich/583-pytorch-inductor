import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mv_OutModule(torch.nn.Module):
    def forward(self, x, vec, out):
        return torch.ops.aten.mv.out(x, vec, out=out)

mod = Torch_Ops_Aten_Mv_OutModule()

x = torch.randn(3)
vec = torch.randn(3)
out = torch.empty(3)

args = (x, vec, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
