import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Roll_OutModule(torch.nn.Module):
    def forward(self, x, shifts, dims, out):
        return torch.ops.aten.roll.out(x, shifts, dims, out=out)

mod = Torch_Ops_Aten_Roll_OutModule()

x = torch.randn(3)
shifts = torch.sym_int(3)
dims = 3
out = torch.empty(3)

args = (x, shifts, dims, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
