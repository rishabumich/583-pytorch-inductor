import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_IndexFill_IntScalarModule(torch.nn.Module):
    def forward(self, x, dim, index, value):
        return torch.ops.aten.index_fill_.int_Scalar(x, dim, index, value)

mod = Torch_Ops_Aten_IndexFill_IntScalarModule()

x = torch.randn(3)
dim = 3
index = torch.randn(3)
value = 1

args = (x, dim, index, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
