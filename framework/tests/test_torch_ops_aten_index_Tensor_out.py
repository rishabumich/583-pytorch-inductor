import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Index_TensorOutModule(torch.nn.Module):
    def forward(self, x, indices, out):
        return torch.ops.aten.index.Tensor_out(x, indices, out=out)

mod = Torch_Ops_Aten_Index_TensorOutModule()

x = torch.randn(3)
indices = torch.randn(3)
out = torch.empty(3)

args = (x, indices, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
