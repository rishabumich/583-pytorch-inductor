import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_ThresholdBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, threshold):
        return torch.ops.aten.threshold_backward(grad_output, x, threshold)

mod = Torch_Ops_Aten_ThresholdBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
threshold = torch.tensor(0)  # Fallback for unknown type Scalar

args = (grad_output, x, threshold,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
