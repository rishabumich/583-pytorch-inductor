import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Cat_NamesOutModule(torch.nn.Module):
    def forward(self, tensors, dim, out):
        return torch.ops.aten.cat.names_out(tensors, dim, out=out)

mod = Torch_Ops_Aten_Cat_NamesOutModule()

tensors = torch.randn(3)
dim = None  # Fallback for unknown type str
out = torch.empty(3)

args = (tensors, dim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
