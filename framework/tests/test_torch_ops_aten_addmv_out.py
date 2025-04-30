import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Addmv_OutModule(torch.nn.Module):
    def forward(self, x, mat, vec, beta, alpha, out):
        return torch.ops.aten.addmv.out(x, mat, vec, beta, alpha, out=out)

mod = Torch_Ops_Aten_Addmv_OutModule()

x = torch.randn(3)
mat = torch.randn(3)
vec = torch.randn(3)
beta = 1
alpha = 1
out = torch.empty(3)

args = (x, mat, vec, beta, alpha, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
