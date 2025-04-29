import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Select_TModule(torch.nn.Module):
    def forward(self, list, idx):
        return torch.ops.aten.select.t(list, idx)

mod = Torch_Ops_Aten_Select_TModule()

list = torch.tensor(0)  # Fallback for unknown type |t[](a)
idx = 3

args = (list, idx,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
