import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HardtanhModule(torch.nn.Module):
    def forward(self, x, min_val, max_val):
        return torch.ops.aten.hardtanh(x, min_val, max_val)

mod = Torch_Ops_Aten_HardtanhModule()

x = torch.randn(3)
min_val = 1
max_val = 1

args = (x, min_val, max_val,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
