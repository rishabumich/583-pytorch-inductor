import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ClampMax_TensorOutModule(torch.nn.Module):
    def forward(self, x, max, out):
        return torch.ops.aten.clamp_max.Tensor_out(x, max, out=out)

mod = Torch_Ops_Aten_ClampMax_TensorOutModule()

x = torch.randn(3)
max = torch.randn(3)
out = torch.empty(3)

args = (x, max, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
