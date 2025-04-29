import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SvdModule(torch.nn.Module):
    def forward(self, x, some, compute_uv):
        return torch.ops.aten.svd(x, some, compute_uv)

mod = Torch_Ops_Aten_SvdModule()

x = torch.randn(3)
some = True
compute_uv = True

args = (x, some, compute_uv,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
