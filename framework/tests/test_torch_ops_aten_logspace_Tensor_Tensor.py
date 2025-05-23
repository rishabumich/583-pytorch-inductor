import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Logspace_TensorTensorModule(torch.nn.Module):
    def forward(self, start, end, steps, base, dtype, layout, device, pin_memory):
        return torch.ops.aten.logspace.Tensor_Tensor(start, end, steps, base, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_Logspace_TensorTensorModule()

start = torch.randn(3)
end = torch.randn(3)
steps = 3
base = 1.0
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True

args = (start, end, steps, base, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
