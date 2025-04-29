import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Normal_FloatTensorModule(torch.nn.Module):
    def forward(self, mean, std, generator):
        return torch.ops.aten.normal.float_Tensor(mean, std, generator)

mod = Torch_Ops_Aten_Normal_FloatTensorModule()

mean = torch.tensor(0)  # Fallback for unknown type |float
std = torch.randn(3)
generator = torch.tensor(0)  # Fallback for unknown type Generator?

args = (mean, std, generator,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
