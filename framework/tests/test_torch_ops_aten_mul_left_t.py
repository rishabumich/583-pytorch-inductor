import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mul_LeftTModule(torch.nn.Module):
    def forward(self, l, n):
        return torch.ops.aten.mul.left_t(l, n)

mod = Torch_Ops_Aten_Mul_LeftTModule()

l = None  # Fallback for unknown type |t[]
n = 3

args = (l, n,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
