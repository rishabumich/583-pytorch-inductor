import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeLayerNormModule(torch.nn.Module):
    def forward(self, input, normalized_shape, weight, bias, eps):
        return torch.ops.aten.native_layer_norm(input, normalized_shape, weight, bias, eps)

mod = Torch_Ops_Aten_NativeLayerNormModule()

input = torch.randn(3)
normalized_shape = torch.tensor(0)  # Fallback for unknown type SymInt[]
weight = torch.randn(3)
bias = torch.randn(3)
eps = 1.0

args = (input, normalized_shape, weight, bias, eps,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
