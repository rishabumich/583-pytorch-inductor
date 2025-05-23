import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Mul_FloatComplexModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten.mul.float_complex(a, b)

mod = Torch_Ops_Aten_Mul_FloatComplexModule()

a = None  # Fallback for unknown type |float
b = complex(1.0, 2.0)

args = (a, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
