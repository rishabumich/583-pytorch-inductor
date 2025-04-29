import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PixelUnshuffleModule(torch.nn.Module):
    def forward(self, x, downscale_factor):
        return torch.ops.aten.pixel_unshuffle(x, downscale_factor)

mod = Torch_Ops_Aten_PixelUnshuffleModule()

x = torch.randn(3)
downscale_factor = 3

args = (x, downscale_factor,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
