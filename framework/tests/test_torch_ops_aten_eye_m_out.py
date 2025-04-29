import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Eye_MOutModule(torch.nn.Module):
    def forward(self, n, m, out):
        return torch.ops.aten.eye.m_out(n, m, out=out)

mod = Torch_Ops_Aten_Eye_MOutModule()

n = torch.tensor(0)  # Fallback for unknown type |SymInt
m = torch.tensor(0)  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (n, m, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
