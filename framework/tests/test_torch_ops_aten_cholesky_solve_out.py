import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_CholeskySolve_OutModule(torch.nn.Module):
    def forward(self, x, input2, upper, out):
        return torch.ops.aten.cholesky_solve.out(x, input2, upper, out=out)

mod = Torch_Ops_Aten_CholeskySolve_OutModule()

x = torch.randn(3)
input2 = torch.randn(3)
upper = True
out = torch.empty(3)

args = (x, input2, upper, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
