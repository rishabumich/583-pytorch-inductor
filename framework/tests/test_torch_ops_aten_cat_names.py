import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Cat_NamesModule(torch.nn.Module):
    def forward(self, tensors, dim):
        return torch.ops.aten.cat.names(tensors, dim)

mod = Torch_Ops_Aten_Cat_NamesModule()

tensors = torch.randn(3)
dim = None  # Fallback for unknown type str

args = (tensors, dim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
