import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EmbeddingModule(torch.nn.Module):
    def forward(self, weight, indices, padding_idx, scale_grad_by_freq, sparse):
        return torch.ops.aten.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

mod = Torch_Ops_Aten_EmbeddingModule()

weight = torch.randn(3)
indices = torch.randn(3)
padding_idx = None  # Fallback for unknown type SymInt
scale_grad_by_freq = True
sparse = True

args = (weight, indices, padding_idx, scale_grad_by_freq, sparse,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
