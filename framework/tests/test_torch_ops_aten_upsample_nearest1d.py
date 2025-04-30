import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleNearest1DModule(torch.nn.Module):
    def forward(self, x, output_size, scales):
        return torch.ops.aten.upsample_nearest1d(x, output_size, scales)

mod = Torch_Ops_Aten_UpsampleNearest1DModule()

x = torch.randn(3)
output_size = torch.sym_int(3)
scales = 1.0

args = (x, output_size, scales,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
