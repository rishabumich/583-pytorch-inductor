import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexPut_OutModule(torch.nn.Module):
    def forward(self, x, indices, values, accumulate, out):
        return torch.ops.aten.index_put.out(x, indices, values, accumulate, out=out)

mod = Torch_Ops_Aten_IndexPut_OutModule()

x = torch.randn(3)
indices = torch.randn(3)
values = torch.randn(3)
accumulate = True
out = torch.empty(3)

args = (x, indices, values, accumulate, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
