import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MultilabelMarginLossForwardModule(torch.nn.Module):
    def forward(self, x, target, reduction):
        return torch.ops.aten.multilabel_margin_loss_forward(x, target, reduction)

mod = Torch_Ops_Aten_MultilabelMarginLossForwardModule()

x = torch.randn(3)
target = torch.randn(3)
reduction = 3

args = (x, target, reduction,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
