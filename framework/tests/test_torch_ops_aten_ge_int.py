import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Ge_IntModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten.ge.int(a, b)

mod = Torch_Ops_Aten_Ge_IntModule()

a = None  # Fallback for unknown type |int
b = 3

args = (a, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
