import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Full_OutModule(torch.nn.Module):
    def forward(self, size, fill_value, out):
        return torch.ops.aten.full.out(size, fill_value, out=out)

mod = Torch_Ops_Aten_Full_OutModule()

size = torch.sym_int(3)
fill_value = 1
out = torch.empty(3)

args = (size, fill_value, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
