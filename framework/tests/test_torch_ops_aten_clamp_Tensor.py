import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Clamp_TensorModule(torch.nn.Module):
    def forward(self, x, min, max):
        return torch.ops.aten.clamp.Tensor(x, min, max)

mod = Torch_Ops_Aten_Clamp_TensorModule()

x = torch.randn(3)
min = torch.randn(3)
max = torch.randn(3)

args = (x, min, max,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
