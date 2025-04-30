import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randint_LowOutModule(torch.nn.Module):
    def forward(self, low, high, size, out):
        return torch.ops.aten.randint.low_out(low, high, size, out=out)

mod = Torch_Ops_Aten_Randint_LowOutModule()

low = None  # Fallback for unknown type |SymInt
high = None  # Fallback for unknown type SymInt
size = torch.sym_int(3)
out = torch.empty(3)

args = (low, high, size, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
