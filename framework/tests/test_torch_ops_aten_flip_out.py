import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Flip_OutModule(torch.nn.Module):
    def forward(self, x, dims, out):
        return torch.ops.aten.flip.out(x, dims, out=out)

mod = Torch_Ops_Aten_Flip_OutModule()

x = torch.randn(3)
dims = torch.tensor(0)  # Fallback for unknown type int[]
out = torch.empty(3)

args = (x, dims, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
