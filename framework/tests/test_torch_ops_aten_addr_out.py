import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Addr_OutModule(torch.nn.Module):
    def forward(self, x, vec1, vec2, beta, alpha, out):
        return torch.ops.aten.addr.out(x, vec1, vec2, beta, alpha, out=out)

mod = Torch_Ops_Aten_Addr_OutModule()

x = torch.randn(3)
vec1 = torch.randn(3)
vec2 = torch.randn(3)
beta = 1
alpha = 1
out = torch.empty(3)

args = (x, vec1, vec2, beta, alpha, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
