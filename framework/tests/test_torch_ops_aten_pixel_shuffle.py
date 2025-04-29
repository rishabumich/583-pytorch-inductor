import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PixelShuffleModule(torch.nn.Module):
    def forward(self, x, upscale_factor):
        return torch.ops.aten.pixel_shuffle(x, upscale_factor)

mod = Torch_Ops_Aten_PixelShuffleModule()

x = torch.randn(3)
upscale_factor = 3

args = (x, upscale_factor,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
