import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CholeskyInverse_OutModule(torch.nn.Module):
    def forward(self, x, upper, out):
        return torch.ops.aten.cholesky_inverse.out(x, upper, out=out)

mod = Torch_Ops_Aten_CholeskyInverse_OutModule()

x = torch.randn(3)
upper = True
out = torch.empty(3)

args = (x, upper, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
