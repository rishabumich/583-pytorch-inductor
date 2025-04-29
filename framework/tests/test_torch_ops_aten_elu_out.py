import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Elu_OutModule(torch.nn.Module):
    def forward(self, x, alpha, scale, input_scale, out):
        return torch.ops.aten.elu.out(x, alpha, scale, input_scale, out=out)

mod = Torch_Ops_Aten_Elu_OutModule()

x = torch.randn(3)
alpha = torch.tensor(0)  # Fallback for unknown type Scalar
scale = torch.tensor(0)  # Fallback for unknown type Scalar
input_scale = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, alpha, scale, input_scale, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
