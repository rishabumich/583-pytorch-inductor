import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UpsampleTrilinear3D_VecModule(torch.nn.Module):
    def forward(self, input, output_size, align_corners, scale_factors):
        return torch.ops.aten.upsample_trilinear3d.vec(input, output_size, align_corners, scale_factors)

mod = Torch_Ops_Aten_UpsampleTrilinear3D_VecModule()

input = torch.randn(3)
output_size = torch.tensor(0)  # Fallback for unknown type SymInt[]?
align_corners = True
scale_factors = torch.tensor(0)  # Fallback for unknown type float[]?

args = (input, output_size, align_corners, scale_factors,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
