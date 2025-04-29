import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mul_TModule(torch.nn.Module):
    def forward(self, l, n):
        return torch.ops.aten.mul_.t(l, n)

mod = Torch_Ops_Aten_Mul_TModule()

l = torch.tensor(0)  # Fallback for unknown type |t[](a!)
n = 3

args = (l, n,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
