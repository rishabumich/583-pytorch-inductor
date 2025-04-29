import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Ormqr_OutModule(torch.nn.Module):
    def forward(self, x, input2, input3, left, transpose, out):
        return torch.ops.aten.ormqr.out(x, input2, input3, left, transpose, out=out)

mod = Torch_Ops_Aten_Ormqr_OutModule()

x = torch.randn(3)
input2 = torch.randn(3)
input3 = torch.randn(3)
left = True
transpose = True
out = torch.empty(3)

args = (x, input2, input3, left, transpose, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
