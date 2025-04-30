import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MaxUnpool3DModule(torch.nn.Module):
    def forward(self, x, indices, output_size, stride, padding):
        return torch.ops.aten.max_unpool3d(x, indices, output_size, stride, padding)

mod = Torch_Ops_Aten_MaxUnpool3DModule()

x = torch.randn(3)
indices = torch.randn(3)
output_size = torch.sym_int(3)
stride = 3
padding = 3

args = (x, indices, output_size, stride, padding,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
