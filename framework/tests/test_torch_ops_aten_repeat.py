import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_RepeatModule(torch.nn.Module):
    def forward(self, x, repeats):
        return torch.ops.aten.repeat(x, repeats)

mod = Torch_Ops_Aten_RepeatModule()

x = torch.randn(3)
repeats = torch.sym_int(3)

args = (x, repeats,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
