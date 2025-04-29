import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Polygamma_OutModule(torch.nn.Module):
    def forward(self, n, x, out):
        return torch.ops.aten.polygamma.out(n, x, out=out)

mod = Torch_Ops_Aten_Polygamma_OutModule()

n = torch.tensor(0)  # Fallback for unknown type |int
x = torch.randn(3)
out = torch.empty(3)

args = (n, x, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
