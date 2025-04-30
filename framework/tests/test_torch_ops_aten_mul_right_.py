import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mul_RightModule(torch.nn.Module):
    def forward(self, n, l):
        return torch.ops.aten.mul.right_(n, l)

mod = Torch_Ops_Aten_Mul_RightModule()

n = None  # Fallback for unknown type |int
l = None  # Fallback for unknown type t[]

args = (n, l,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
