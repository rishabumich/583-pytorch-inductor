import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Clip_OutModule(torch.nn.Module):
    def forward(self, x, min, max, out):
        return torch.ops.aten.clip.out(x, min, max, out=out)

mod = Torch_Ops_Aten_Clip_OutModule()

x = torch.randn(3)
min = torch.tensor(0)  # Fallback for unknown type Scalar?
max = torch.tensor(0)  # Fallback for unknown type Scalar?
out = torch.empty(3)

args = (x, min, max, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
