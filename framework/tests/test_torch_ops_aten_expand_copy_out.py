import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ExpandCopy_OutModule(torch.nn.Module):
    def forward(self, x, size, implicit, out):
        return torch.ops.aten.expand_copy.out(x, size, implicit, out=out)

mod = Torch_Ops_Aten_ExpandCopy_OutModule()

x = torch.randn(3)
size = torch.sym_int(3)
implicit = True
out = torch.empty(3)

args = (x, size, implicit, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
