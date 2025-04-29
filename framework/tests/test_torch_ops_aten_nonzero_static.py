import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NonzeroStaticModule(torch.nn.Module):
    def forward(self, x, size, fill_value):
        return torch.ops.aten.nonzero_static(x, size, fill_value)

mod = Torch_Ops_Aten_NonzeroStaticModule()

x = torch.randn(3)
size = 3
fill_value = 3

args = (x, size, fill_value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
