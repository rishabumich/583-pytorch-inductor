import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeLayerNormBackwardModule(torch.nn.Module):
    def forward(self, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask):
        return torch.ops.aten.native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask)

mod = Torch_Ops_Aten_NativeLayerNormBackwardModule()

grad_out = torch.randn(3)
input = torch.randn(3)
normalized_shape = torch.tensor(0)  # Fallback for unknown type SymInt[]
mean = torch.randn(3)
rstd = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
output_mask = torch.tensor(0)  # Fallback for unknown type bool[3]

args = (grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
