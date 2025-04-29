import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ClampMin_TensorOutModule(torch.nn.Module):
    def forward(self, x, min, out):
        return torch.ops.aten.clamp_min.Tensor_out(x, min, out=out)

mod = Torch_Ops_Aten_ClampMin_TensorOutModule()

x = torch.randn(3)
min = torch.randn(3)
out = torch.empty(3)

args = (x, min, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
