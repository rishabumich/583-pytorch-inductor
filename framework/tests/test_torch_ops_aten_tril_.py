import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_TrilModule(torch.nn.Module):
    def forward(self, x, diagonal):
        return torch.ops.aten.tril_(x, diagonal)

mod = Torch_Ops_Aten_TrilModule()

x = torch.randn(3)
diagonal = 3

args = (x, diagonal,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
