import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Frexp_TensorOutModule(torch.nn.Module):
    def forward(self, x, mantissa, exponent):
        return torch.ops.aten.frexp.Tensor_out(x, mantissa, exponent)

mod = Torch_Ops_Aten_Frexp_TensorOutModule()

x = torch.randn(3)
mantissa = torch.randn(3)
exponent = torch.randn(3)

args = (x, mantissa, exponent,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
