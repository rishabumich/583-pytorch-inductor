import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_NativeLayerNormBackward_OutModule(torch.nn.Module):
    def forward(self, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, out0, out1, out2):
        return torch.ops.aten.native_layer_norm_backward.out(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, out0, out1, out2)

mod = Torch_Ops_Aten_NativeLayerNormBackward_OutModule()

grad_out = torch.randn(3)
input = torch.randn(3)
normalized_shape = torch.sym_int(3)
mean = torch.randn(3)
rstd = torch.randn(3)
weight = torch.randn(3)
bias = torch.randn(3)
output_mask = True
out0 = torch.randn(3)
out1 = torch.randn(3)
out2 = torch.randn(3)

args = (grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, out0, out1, out2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
