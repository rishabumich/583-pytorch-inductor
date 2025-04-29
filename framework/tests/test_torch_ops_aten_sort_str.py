import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Sort_StrModule(torch.nn.Module):
    def forward(self, x, reverse):
        return torch.ops.aten.sort.str(x, reverse)

mod = Torch_Ops_Aten_Sort_StrModule()

x = torch.tensor(0)  # Fallback for unknown type |str[](a!)
reverse = True

args = (x, reverse,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
