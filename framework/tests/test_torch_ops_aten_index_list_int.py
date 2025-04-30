import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Index_ListIntModule(torch.nn.Module):
    def forward(self, x, el):
        return torch.ops.aten.index.list_int(x, el)

mod = Torch_Ops_Aten_Index_ListIntModule()

x = 3
el = 3

args = (x, el,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
