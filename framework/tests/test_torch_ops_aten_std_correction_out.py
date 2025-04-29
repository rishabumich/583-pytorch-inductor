import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Std_CorrectionOutModule(torch.nn.Module):
    def forward(self, x, dim, correction, keepdim, out):
        return torch.ops.aten.std.correction_out(x, dim, correction, keepdim, out=out)

mod = Torch_Ops_Aten_Std_CorrectionOutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type int[1]?
correction = torch.tensor(0)  # Fallback for unknown type Scalar?
keepdim = True
out = torch.empty(3)

args = (x, dim, correction, keepdim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
