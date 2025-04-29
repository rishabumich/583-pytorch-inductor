import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Index_ListFloatModule(torch.nn.Module):
    def forward(self, x, el):
        return torch.ops.aten.index.list_float(x, el)

mod = Torch_Ops_Aten_Index_ListFloatModule()

x = torch.tensor(0)  # Fallback for unknown type |float[]
el = 1.0

args = (x, el,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
