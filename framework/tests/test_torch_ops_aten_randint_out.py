import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randint_OutModule(torch.nn.Module):
    def forward(self, high, size, out):
        return torch.ops.aten.randint.out(high, size, out=out)

mod = Torch_Ops_Aten_Randint_OutModule()

high = torch.tensor(0)  # Fallback for unknown type |SymInt
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
out = torch.empty(3)

args = (high, size, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
