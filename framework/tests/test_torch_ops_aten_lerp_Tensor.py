import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Lerp_TensorModule(torch.nn.Module):
    def forward(self, x, end, weight):
        return torch.ops.aten.lerp.Tensor(x, end, weight)

mod = Torch_Ops_Aten_Lerp_TensorModule()

x = torch.randn(3)
end = torch.randn(3)
weight = torch.randn(3)

args = (x, end, weight,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
