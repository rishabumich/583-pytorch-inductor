import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UnfoldCopyModule(torch.nn.Module):
    def forward(self, x, dimension, size, step):
        return torch.ops.aten.unfold_copy(x, dimension, size, step)

mod = Torch_Ops_Aten_UnfoldCopyModule()

x = torch.randn(3)
dimension = 3
size = 3
step = 3

args = (x, dimension, size, step,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
