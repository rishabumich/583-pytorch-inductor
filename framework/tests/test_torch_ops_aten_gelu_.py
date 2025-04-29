import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_GeluModule(torch.nn.Module):
    def forward(self, x, approximate):
        return torch.ops.aten.gelu_(x, approximate)

mod = Torch_Ops_Aten_GeluModule()

x = torch.randn(3)
approximate = torch.tensor(0)  # Fallback for unknown type str

args = (x, approximate,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
