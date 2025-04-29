import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexCopy_OutModule(torch.nn.Module):
    def forward(self, x, dim, index, source, out):
        return torch.ops.aten.index_copy.out(x, dim, index, source, out=out)

mod = Torch_Ops_Aten_IndexCopy_OutModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
source = torch.randn(3)
out = torch.empty(3)

args = (x, dim, index, source, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
