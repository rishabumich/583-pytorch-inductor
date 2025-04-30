import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AdaptiveMaxPool2D_OutModule(torch.nn.Module):
    def forward(self, x, output_size, out, indices):
        return torch.ops.aten.adaptive_max_pool2d.out(x, output_size, out=out, indices)

mod = Torch_Ops_Aten_AdaptiveMaxPool2D_OutModule()

x = torch.randn(3)
output_size = 3
out = torch.empty(3)
indices = torch.randn(3)

args = (x, output_size, out, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
