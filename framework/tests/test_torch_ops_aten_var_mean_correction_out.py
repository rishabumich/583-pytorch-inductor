import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_VarMean_CorrectionOutModule(torch.nn.Module):
    def forward(self, x, dim, correction, keepdim, out0, out1):
        return torch.ops.aten.var_mean.correction_out(x, dim, correction, keepdim, out0, out1)

mod = Torch_Ops_Aten_VarMean_CorrectionOutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
correction = torch.tensor(0)  # Fallback for unknown type Scalar?
keepdim = True
out0 = torch.randn(3)
out1 = torch.randn(3)

args = (x, dim, correction, keepdim, out0, out1,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
