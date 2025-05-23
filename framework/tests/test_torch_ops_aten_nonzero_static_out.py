import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NonzeroStatic_OutModule(torch.nn.Module):
    def forward(self, x, size, fill_value, out):
        return torch.ops.aten.nonzero_static.out(x, size, fill_value, out=out)

mod = Torch_Ops_Aten_NonzeroStatic_OutModule()

x = torch.randn(3)
size = 3
fill_value = 3
out = torch.empty(3)

args = (x, size, fill_value, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
