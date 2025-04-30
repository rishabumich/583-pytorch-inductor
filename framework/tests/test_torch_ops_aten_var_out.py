import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Var_OutModule(torch.nn.Module):
    def forward(self, x, dim, unbiased, keepdim, out):
        return torch.ops.aten.var.out(x, dim, unbiased, keepdim, out=out)

mod = Torch_Ops_Aten_Var_OutModule()

x = torch.randn(3)
dim = 3
unbiased = True
keepdim = True
out = torch.empty(3)

args = (x, dim, unbiased, keepdim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
