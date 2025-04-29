import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Eq_DeviceModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.aten.eq.device(a, b)

mod = Torch_Ops_Aten_Eq_DeviceModule()

a = torch.tensor(0)  # Fallback for unknown type |Device
b = torch.tensor(0)  # Fallback for unknown type Device

args = (a, b,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
