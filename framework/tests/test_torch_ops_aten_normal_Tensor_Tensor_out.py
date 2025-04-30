import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Normal_TensorTensorOutModule(torch.nn.Module):
    def forward(self, mean, std, generator, out):
        return torch.ops.aten.normal.Tensor_Tensor_out(mean, std, generator, out=out)

mod = Torch_Ops_Aten_Normal_TensorTensorOutModule()

mean = torch.randn(3)
std = torch.randn(3)
generator = None  # Fallback for unknown type Generator?
out = torch.empty(3)

args = (mean, std, generator, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
