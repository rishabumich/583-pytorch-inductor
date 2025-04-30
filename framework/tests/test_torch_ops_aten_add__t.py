import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Add_TModule(torch.nn.Module):
    def forward(self, x, b):
        return torch.ops.aten.add_.t(x, b)

mod = Torch_Ops_Aten_Add_TModule()

x = None  # Fallback for unknown type |t[](a!)
b = None  # Fallback for unknown type t[]

args = (x, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
