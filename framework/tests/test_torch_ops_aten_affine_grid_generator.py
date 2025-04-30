import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AffineGridGeneratorModule(torch.nn.Module):
    def forward(self, theta, size, align_corners):
        return torch.ops.aten.affine_grid_generator(theta, size, align_corners)

mod = Torch_Ops_Aten_AffineGridGeneratorModule()

theta = torch.randn(3)
size = torch.sym_int(3)
align_corners = True

args = (theta, size, align_corners,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
