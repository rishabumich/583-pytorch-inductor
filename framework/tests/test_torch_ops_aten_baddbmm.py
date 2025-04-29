import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_BaddbmmModule(torch.nn.Module):
    def forward(self, x, batch1, batch2, beta, alpha):
        return torch.ops.aten.baddbmm(x, batch1, batch2, beta, alpha)

mod = Torch_Ops_Aten_BaddbmmModule()

x = torch.randn(3)
batch1 = torch.randn(3)
batch2 = torch.randn(3)
beta = torch.tensor(0)  # Fallback for unknown type Scalar
alpha = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, batch1, batch2, beta, alpha,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
