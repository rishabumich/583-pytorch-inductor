import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SliceScatter_OutModule(torch.nn.Module):
    def forward(self, x, src, dim, start, end, step, out):
        return torch.ops.aten.slice_scatter.out(x, src, dim, start, end, step, out=out)

mod = Torch_Ops_Aten_SliceScatter_OutModule()

x = torch.randn(3)
src = torch.randn(3)
dim = 3
start = torch.tensor(0)  # Fallback for unknown type SymInt?
end = torch.tensor(0)  # Fallback for unknown type SymInt?
step = torch.tensor(0)  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (x, src, dim, start, end, step, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
