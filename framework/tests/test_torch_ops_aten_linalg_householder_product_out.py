import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_LinalgHouseholderProduct_OutModule(torch.nn.Module):
    def forward(self, input, tau, out):
        return torch.ops.aten.linalg_householder_product.out(input, tau, out=out)

mod = Torch_Ops_Aten_LinalgHouseholderProduct_OutModule()

input = torch.randn(3)
tau = torch.randn(3)
out = torch.empty(3)

args = (input, tau, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
