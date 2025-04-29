import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Svd_UModule(torch.nn.Module):
    def forward(self, x, some, compute_uv, U, S, V):
        return torch.ops.aten.svd.U(x, some, compute_uv, U, S, V)

mod = Torch_Ops_Aten_Svd_UModule()

x = torch.randn(3)
some = True
compute_uv = True
U = torch.randn(3)
S = torch.randn(3)
V = torch.randn(3)

args = (x, some, compute_uv, U, S, V,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
