import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LogitModule(torch.nn.Module):
    def forward(self, x, eps):
        return torch.ops.aten.logit(x, eps)

mod = Torch_Ops_Aten_LogitModule()

x = torch.randn(3)
eps = 1.0

args = (x, eps,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
