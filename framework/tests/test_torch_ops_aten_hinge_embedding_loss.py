import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HingeEmbeddingLossModule(torch.nn.Module):
    def forward(self, x, target, margin, reduction):
        return torch.ops.aten.hinge_embedding_loss(x, target, margin, reduction)

mod = Torch_Ops_Aten_HingeEmbeddingLossModule()

x = torch.randn(3)
target = torch.randn(3)
margin = 1.0
reduction = 3

args = (x, target, margin, reduction,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
