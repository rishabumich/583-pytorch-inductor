import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Fill_ScalarOutModule(torch.nn.Module):
    def forward(self, x, value, out):
        return torch.ops.aten.fill.Scalar_out(x, value, out=out)

mod = Torch_Ops_Aten_Fill_ScalarOutModule()

x = torch.randn(3)
value = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, value, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
