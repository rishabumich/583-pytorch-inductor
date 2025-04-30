import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Linspace_OutModule(torch.nn.Module):
    def forward(self, start, end, steps, out):
        return torch.ops.aten.linspace.out(start, end, steps, out=out)

mod = Torch_Ops_Aten_Linspace_OutModule()

start = None  # Fallback for unknown type |Scalar
end = 1
steps = 3
out = torch.empty(3)

args = (start, end, steps, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
