import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Repeat_OutModule(torch.nn.Module):
    def forward(self, x, repeats, out):
        return torch.ops.aten.repeat.out(x, repeats, out=out)

mod = Torch_Ops_Aten_Repeat_OutModule()

x = torch.randn(3)
repeats = torch.tensor(0)  # Fallback for unknown type SymInt[]
out = torch.empty(3)

args = (x, repeats, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
