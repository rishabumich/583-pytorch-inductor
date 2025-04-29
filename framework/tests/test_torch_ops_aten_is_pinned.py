import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IsPinnedModule(torch.nn.Module):
    def forward(self, x, device):
        return torch.ops.aten.is_pinned(x, device)

mod = Torch_Ops_Aten_IsPinnedModule()

x = torch.randn(3)
device = torch.tensor(0)  # Fallback for unknown type Device?

args = (x, device,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
