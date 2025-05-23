import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_View_DtypeModule(torch.nn.Module):
    def forward(self, x, dtype):
        return torch.ops.aten.view.dtype(x, dtype)

mod = Torch_Ops_Aten_View_DtypeModule()

x = torch.randn(3)
dtype = None  # Fallback for unknown type ScalarType

args = (x, dtype,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
