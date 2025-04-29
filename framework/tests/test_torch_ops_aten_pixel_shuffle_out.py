import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_PixelShuffle_OutModule(torch.nn.Module):
    def forward(self, x, upscale_factor, out):
        return torch.ops.aten.pixel_shuffle.out(x, upscale_factor, out=out)

mod = Torch_Ops_Aten_PixelShuffle_OutModule()

x = torch.randn(3)
upscale_factor = 3
out = torch.empty(3)

args = (x, upscale_factor, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
