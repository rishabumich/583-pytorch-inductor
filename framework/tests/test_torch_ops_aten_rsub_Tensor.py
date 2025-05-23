import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Rsub_TensorModule(torch.nn.Module):
    def forward(self, x, other, alpha):
        return torch.ops.aten.rsub.Tensor(x, other, alpha)

mod = Torch_Ops_Aten_Rsub_TensorModule()

x = torch.randn(3)
other = torch.randn(3)
alpha = 1

args = (x, other, alpha,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
