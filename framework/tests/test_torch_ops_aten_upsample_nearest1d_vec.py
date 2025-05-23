import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleNearest1D_VecModule(torch.nn.Module):
    def forward(self, input, output_size, scale_factors):
        return torch.ops.aten.upsample_nearest1d.vec(input, output_size, scale_factors)

mod = Torch_Ops_Aten_UpsampleNearest1D_VecModule()

input = torch.randn(3)
output_size = torch.sym_int(3)
scale_factors = 1.0

args = (input, output_size, scale_factors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
