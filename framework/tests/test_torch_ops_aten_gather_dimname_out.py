import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Gather_DimnameOutModule(torch.nn.Module):
    def forward(self, x, dim, index, sparse_grad, out):
        return torch.ops.aten.gather.dimname_out(x, dim, index, sparse_grad, out=out)

mod = Torch_Ops_Aten_Gather_DimnameOutModule()

x = torch.randn(3)
dim = torch.tensor(0)  # Fallback for unknown type str
index = torch.randn(3)
sparse_grad = True
out = torch.empty(3)

args = (x, dim, index, sparse_grad, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
