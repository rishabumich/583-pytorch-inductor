import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UnsafeChunkModule(torch.nn.Module):
    def forward(self, x, chunks, dim):
        return torch.ops.aten.unsafe_chunk(x, chunks, dim)

mod = Torch_Ops_Aten_UnsafeChunkModule()

x = torch.randn(3)
chunks = 3
dim = 3

args = (x, chunks, dim,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
