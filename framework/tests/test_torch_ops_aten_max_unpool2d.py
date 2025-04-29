import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaxUnpool2DModule(torch.nn.Module):
    def forward(self, x, indices, output_size):
        return torch.ops.aten.max_unpool2d(x, indices, output_size)

mod = Torch_Ops_Aten_MaxUnpool2DModule()

x = torch.randn(3)
indices = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt[2]

args = (x, indices, output_size,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
