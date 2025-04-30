import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Col2Im_OutModule(torch.nn.Module):
    def forward(self, x, output_size, kernel_size, dilation, padding, stride, out):
        return torch.ops.aten.col2im.out(x, output_size, kernel_size, dilation, padding, stride, out=out)

mod = Torch_Ops_Aten_Col2Im_OutModule()

x = torch.randn(3)
output_size = torch.sym_int(3)
kernel_size = 3
dilation = 3
padding = 3
stride = 3
out = torch.empty(3)

args = (x, output_size, kernel_size, dilation, padding, stride, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
