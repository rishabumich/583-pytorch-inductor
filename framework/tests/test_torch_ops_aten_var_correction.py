import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Var_CorrectionModule(torch.nn.Module):
    def forward(self, x, dim, correction, keepdim):
        return torch.ops.aten.var.correction(x, dim, correction, keepdim)

mod = Torch_Ops_Aten_Var_CorrectionModule()

x = torch.randn(3)
dim = 3
correction = 1
keepdim = True

args = (x, dim, correction, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
