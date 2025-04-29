import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_MultilabelMarginLossForward_OutputModule(torch.nn.Module):
    def forward(self, x, target, reduction, output, is_target):
        return torch.ops.aten.multilabel_margin_loss_forward.output(x, target, reduction, output, is_target)

mod = Torch_Ops_Aten_MultilabelMarginLossForward_OutputModule()

x = torch.randn(3)
target = torch.randn(3)
reduction = 3
output = torch.randn(3)
is_target = torch.randn(3)

args = (x, target, reduction, output, is_target,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
