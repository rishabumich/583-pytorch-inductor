import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Arange_StartStepModule(torch.nn.Module):
    def forward(self, start, end, step, dtype, layout, device, pin_memory):
        return torch.ops.aten.arange.start_step(start, end, step, dtype, layout, device, pin_memory)

mod = Torch_Ops_Aten_Arange_StartStepModule()

start = None  # Fallback for unknown type |Scalar
end = 1
step = 1
dtype = None  # Fallback for unknown type ScalarType?
layout = None  # Fallback for unknown type Layout?
device = None  # Fallback for unknown type Device?
pin_memory = True

args = (start, end, step, dtype, layout, device, pin_memory,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
