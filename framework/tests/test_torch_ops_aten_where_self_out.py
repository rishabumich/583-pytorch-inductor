import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Where_SelfOutModule(torch.nn.Module):
    def forward(self, condition, x, other, out):
        return torch.ops.aten.where.self_out(condition, x, other, out=out)

mod = Torch_Ops_Aten_Where_SelfOutModule()

condition = torch.randn(3)
x = torch.randn(3)
other = torch.randn(3)
out = torch.empty(3)

args = (condition, x, other, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
