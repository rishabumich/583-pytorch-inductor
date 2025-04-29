import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HardshrinkModule(torch.nn.Module):
    def forward(self, x, lambd):
        return torch.ops.aten.hardshrink(x, lambd)

mod = Torch_Ops_Aten_HardshrinkModule()

x = torch.randn(3)
lambd = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, lambd,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
