import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Ones_OutModule(torch.nn.Module):
    def forward(self, size, out):
        return torch.ops.aten.ones.out(size, out=out)

mod = Torch_Ops_Aten_Ones_OutModule()

size = torch.tensor(0)  # Fallback for unknown type |SymInt[]
out = torch.empty(3)

args = (size, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
