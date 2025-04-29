import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Fill_TensorModule(torch.nn.Module):
    def forward(self, x, value):
        return torch.ops.aten.fill_.Tensor(x, value)

mod = Torch_Ops_Aten_Fill_TensorModule()

x = torch.randn(3)
value = torch.randn(3)

args = (x, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
