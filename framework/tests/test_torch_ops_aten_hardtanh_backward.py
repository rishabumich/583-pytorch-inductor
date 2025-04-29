import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_HardtanhBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, min_val, max_val):
        return torch.ops.aten.hardtanh_backward(grad_output, x, min_val, max_val)

mod = Torch_Ops_Aten_HardtanhBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
min_val = torch.tensor(0)  # Fallback for unknown type Scalar
max_val = torch.tensor(0)  # Fallback for unknown type Scalar

args = (grad_output, x, min_val, max_val,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
