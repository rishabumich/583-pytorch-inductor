import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SoftplusModule(torch.nn.Module):
    def forward(self, x, beta, threshold):
        return torch.ops.aten.softplus(x, beta, threshold)

mod = Torch_Ops_Aten_SoftplusModule()

x = torch.randn(3)
beta = torch.tensor(0)  # Fallback for unknown type Scalar
threshold = torch.tensor(0)  # Fallback for unknown type Scalar

args = (x, beta, threshold,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
