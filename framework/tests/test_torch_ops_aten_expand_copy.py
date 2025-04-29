import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ExpandCopyModule(torch.nn.Module):
    def forward(self, x, size, implicit):
        return torch.ops.aten.expand_copy(x, size, implicit)

mod = Torch_Ops_Aten_ExpandCopyModule()

x = torch.randn(3)
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
implicit = True

args = (x, size, implicit,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
