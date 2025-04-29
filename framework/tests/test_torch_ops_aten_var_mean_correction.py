import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_VarMean_CorrectionModule(torch.nn.Module):
    def forward(self, x, dim, correction, keepdim):
        return torch.ops.aten.var_mean.correction(x, dim, correction, keepdim)

mod = Torch_Ops_Aten_VarMean_CorrectionModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
correction = torch.tensor(0)  # Fallback for unknown type Scalar?
keepdim = True

args = (x, dim, correction, keepdim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
