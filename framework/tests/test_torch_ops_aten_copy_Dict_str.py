import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Copy_DictStrModule(torch.nn.Module):
    def forward(self, t):
        return torch.ops.aten.copy.Dict_str(t)

mod = Torch_Ops_Aten_Copy_DictStrModule()

t = torch.tensor(0)  # Fallback for unknown type |Dict(str

args = (t,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
