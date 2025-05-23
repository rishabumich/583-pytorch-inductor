import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CholeskySolveModule(torch.nn.Module):
    def forward(self, x, input2, upper):
        return torch.ops.aten.cholesky_solve(x, input2, upper)

mod = Torch_Ops_Aten_CholeskySolveModule()

x = torch.randn(3)
input2 = torch.randn(3)
upper = True

args = (x, input2, upper,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
