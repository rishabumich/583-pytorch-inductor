import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Randn_NamesOutModule(torch.nn.Module):
    def forward(self, size, names, out):
        return torch.ops.aten.randn.names_out(size, names, out=out)

mod = Torch_Ops_Aten_Randn_NamesOutModule()

size = torch.tensor(0)  # Fallback for unknown type |SymInt[]
names = torch.tensor(0)  # Fallback for unknown type str[]?
out = torch.empty(3)

args = (size, names, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
