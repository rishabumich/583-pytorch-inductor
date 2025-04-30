import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NewEmpty_OutModule(torch.nn.Module):
    def forward(self, x, size, out):
        return torch.ops.aten.new_empty.out(x, size, out=out)

mod = Torch_Ops_Aten_NewEmpty_OutModule()

x = torch.randn(3)
size = torch.sym_int(3)
out = torch.empty(3)

args = (x, size, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
