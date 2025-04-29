import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_DotModule(torch.nn.Module):
    def forward(self, x, tensor):
        return torch.ops.aten.dot(x, tensor)

mod = Torch_Ops_Aten_DotModule()

x = torch.randn(3)
tensor = torch.randn(3)

args = (x, tensor,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
