import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EqModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten.eq(a, b)

mod = Torch_Ops_Aten_EqModule()

a = torch.tensor(0)  # Fallback for unknown type |Scalar
b = torch.tensor(0)  # Fallback for unknown type Scalar

args = (a, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
