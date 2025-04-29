import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ReshapeModule(torch.nn.Module):
    def forward(self, x, shape):
        return torch.ops.aten.reshape(x, shape)

mod = Torch_Ops_Aten_ReshapeModule()

x = torch.randn(3)
shape = torch.tensor(0)  # Fallback for unknown type SymInt[]

args = (x, shape,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
