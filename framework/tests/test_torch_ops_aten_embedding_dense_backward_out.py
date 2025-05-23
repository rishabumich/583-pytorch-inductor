import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EmbeddingDenseBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out):
        return torch.ops.aten.embedding_dense_backward.out(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out=out)

mod = Torch_Ops_Aten_EmbeddingDenseBackward_OutModule()

grad_output = torch.randn(3)
indices = torch.randn(3)
num_weights = None  # Fallback for unknown type SymInt
padding_idx = None  # Fallback for unknown type SymInt
scale_grad_by_freq = True
out = torch.empty(3)

args = (grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
