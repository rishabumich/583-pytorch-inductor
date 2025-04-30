import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ViewModule(torch.nn.Module):
    def forward(self, x, size):
        return torch.ops.aten.view(x, size)

mod = Torch_Ops_Aten_ViewModule()

x = torch.randn(3)
size = torch.sym_int(3)

args = (x, size,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
