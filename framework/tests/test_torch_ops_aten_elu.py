import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EluModule(torch.nn.Module):
    def forward(self, x, alpha, scale, input_scale):
        return torch.ops.aten.elu(x, alpha, scale, input_scale)

mod = Torch_Ops_Aten_EluModule()

x = torch.randn(3)
alpha = 1
scale = 1
input_scale = 1

args = (x, alpha, scale, input_scale,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
