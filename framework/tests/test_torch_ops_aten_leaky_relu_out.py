import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LeakyRelu_OutModule(torch.nn.Module):
    def forward(self, x, negative_slope, out):
        return torch.ops.aten.leaky_relu.out(x, negative_slope, out=out)

mod = Torch_Ops_Aten_LeakyRelu_OutModule()

x = torch.randn(3)
negative_slope = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, negative_slope, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
