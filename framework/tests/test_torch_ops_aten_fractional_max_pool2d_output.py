import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_FractionalMaxPool2D_OutputModule(torch.nn.Module):
    def forward(self, x, kernel_size, output_size, random_samples, output, indices):
        return torch.ops.aten.fractional_max_pool2d.output(x, kernel_size, output_size, random_samples, output, indices)

mod = Torch_Ops_Aten_FractionalMaxPool2D_OutputModule()

x = torch.randn(3)
kernel_size = 3
output_size = 3
random_samples = torch.randn(3)
output = torch.randn(3)
indices = torch.randn(3)

args = (x, kernel_size, output_size, random_samples, output, indices,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
