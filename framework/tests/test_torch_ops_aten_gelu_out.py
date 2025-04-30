import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Gelu_OutModule(torch.nn.Module):
    def forward(self, x, approximate, out):
        return torch.ops.aten.gelu.out(x, approximate, out=out)

mod = Torch_Ops_Aten_Gelu_OutModule()

x = torch.randn(3)
approximate = None  # Fallback for unknown type str
out = torch.empty(3)

args = (x, approximate, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
