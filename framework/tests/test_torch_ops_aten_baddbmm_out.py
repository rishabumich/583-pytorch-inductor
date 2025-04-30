import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Baddbmm_OutModule(torch.nn.Module):
    def forward(self, x, batch1, batch2, beta, alpha, out):
        return torch.ops.aten.baddbmm.out(x, batch1, batch2, beta, alpha, out=out)

mod = Torch_Ops_Aten_Baddbmm_OutModule()

x = torch.randn(3)
batch1 = torch.randn(3)
batch2 = torch.randn(3)
beta = 1
alpha = 1
out = torch.empty(3)

args = (x, batch1, batch2, beta, alpha, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
