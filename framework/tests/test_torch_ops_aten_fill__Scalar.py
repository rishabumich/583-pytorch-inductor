import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Fill_ScalarModule(torch.nn.Module):
    def forward(self, x, value):
        return torch.ops.aten.fill_.Scalar(x, value)

mod = Torch_Ops_Aten_Fill_ScalarModule()

x = torch.randn(3)
value = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
