import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Softplus_OutModule(torch.nn.Module):
    def forward(self, x, beta, threshold, out):
        return torch.ops.aten.softplus.out(x, beta, threshold, out=out)

mod = Torch_Ops_Aten_Softplus_OutModule()

x = torch.randn(3)
beta = torch.tensor(0)  # Fallback for unknown type Scalar
threshold = torch.tensor(0)  # Fallback for unknown type Scalar
out = torch.empty(3)

args = (x, beta, threshold, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
