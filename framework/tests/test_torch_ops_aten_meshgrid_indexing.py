import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Meshgrid_IndexingModule(torch.nn.Module):
    def forward(self, tensors, indexing):
        return torch.ops.aten.meshgrid.indexing(tensors, indexing)

mod = Torch_Ops_Aten_Meshgrid_IndexingModule()

tensors = torch.randn(3)
indexing = None  # Fallback for unknown type str

args = (tensors, indexing,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
