import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Gt_ScalarModule(torch.nn.Module):
    def forward(self, x, other):
        return torch.ops.aten.gt.Scalar(x, other)

mod = Torch_Ops_Aten_Gt_ScalarModule()

x = torch.randn(3)
other = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, other,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
