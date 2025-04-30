import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sub_IntComplexModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten.sub.int_complex(a, b)

mod = Torch_Ops_Aten_Sub_IntComplexModule()

a = None  # Fallback for unknown type |int
b = complex(1.0, 2.0)

args = (a, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
