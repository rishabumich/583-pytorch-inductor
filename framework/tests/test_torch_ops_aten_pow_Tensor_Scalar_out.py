import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Pow_TensorScalarOutModule(torch.nn.Module):
    def forward(self, x, exponent, out):
        return torch.ops.aten.pow.Tensor_Scalar_out(x, exponent, out=out)

mod = Torch_Ops_Aten_Pow_TensorScalarOutModule()

x = torch.randn(3)
exponent = 1
out = torch.empty(3)

args = (x, exponent, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
