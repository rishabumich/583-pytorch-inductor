import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Lerp_TensorOutModule(torch.nn.Module):
    def forward(self, x, end, weight, out):
        return torch.ops.aten.lerp.Tensor_out(x, end, weight, out=out)

mod = Torch_Ops_Aten_Lerp_TensorOutModule()

x = torch.randn(3)
end = torch.randn(3)
weight = torch.randn(3)
out = torch.empty(3)

args = (x, end, weight, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
