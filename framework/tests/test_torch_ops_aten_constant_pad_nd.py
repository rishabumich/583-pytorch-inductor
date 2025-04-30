import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ConstantPadNdModule(torch.nn.Module):
    def forward(self, x, pad, value):
        return torch.ops.aten.constant_pad_nd(x, pad, value)

mod = Torch_Ops_Aten_ConstantPadNdModule()

x = torch.randn(3)
pad = torch.sym_int(3)
value = 1

args = (x, pad, value,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
