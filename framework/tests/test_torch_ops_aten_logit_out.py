import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Logit_OutModule(torch.nn.Module):
    def forward(self, x, eps, out):
        return torch.ops.aten.logit.out(x, eps, out=out)

mod = Torch_Ops_Aten_Logit_OutModule()

x = torch.randn(3)
eps = torch.tensor(0)  # Fallback for unknown type float?
out = torch.empty(3)

args = (x, eps, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
