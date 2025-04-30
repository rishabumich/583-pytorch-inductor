import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Embedding_OutModule(torch.nn.Module):
    def forward(self, weight, indices, padding_idx, scale_grad_by_freq, sparse, out):
        return torch.ops.aten.embedding.out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out=out)

mod = Torch_Ops_Aten_Embedding_OutModule()

weight = torch.randn(3)
indices = torch.randn(3)
padding_idx = None  # Fallback for unknown type SymInt
scale_grad_by_freq = True
sparse = True
out = torch.empty(3)

args = (weight, indices, padding_idx, scale_grad_by_freq, sparse, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
