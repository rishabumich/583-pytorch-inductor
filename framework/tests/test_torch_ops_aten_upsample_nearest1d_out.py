import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleNearest1D_OutModule(torch.nn.Module):
    def forward(self, x, output_size, scales, out):
        return torch.ops.aten.upsample_nearest1d.out(x, output_size, scales, out=out)

mod = Torch_Ops_Aten_UpsampleNearest1D_OutModule()

x = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt[1]
scales = torch.tensor(0)  # Fallback for unknown type float?
out = torch.empty(3)

args = (x, output_size, scales, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
