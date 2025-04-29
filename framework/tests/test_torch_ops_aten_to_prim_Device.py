import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_To_PrimDeviceModule(torch.nn.Module):
    def forward(self, x, device, dtype, non_blocking, copy):
        return torch.ops.aten.to.prim_Device(x, device, dtype, non_blocking, copy)

mod = Torch_Ops_Aten_To_PrimDeviceModule()

x = torch.randn(3)
device = torch.tensor(0)  # Fallback for unknown type Device?
dtype = torch.tensor(0)  # Fallback for unknown type int?
non_blocking = True
copy = True

args = (x, device, dtype, non_blocking, copy,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
