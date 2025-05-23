import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Normal_TensorFloatModule(torch.nn.Module):
    def forward(self, mean, std, generator):
        return torch.ops.aten.normal.Tensor_float(mean, std, generator)

mod = Torch_Ops_Aten_Normal_TensorFloatModule()

mean = torch.randn(3)
std = 1.0
generator = None  # Fallback for unknown type Generator?

args = (mean, std, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
