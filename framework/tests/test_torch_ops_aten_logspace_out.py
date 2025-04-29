import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Logspace_OutModule(torch.nn.Module):
    def forward(self, start, end, steps, base, out):
        return torch.ops.aten.logspace.out(start, end, steps, base, out=out)

mod = Torch_Ops_Aten_Logspace_OutModule()

start = torch.tensor(0)  # Fallback for unknown type |Scalar
end = torch.tensor(0)  # Fallback for unknown type Scalar
steps = 3
base = 1.0
out = torch.empty(3)

args = (start, end, steps, base, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
