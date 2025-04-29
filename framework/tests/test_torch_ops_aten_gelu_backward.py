import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_GeluBackwardModule(torch.nn.Module):
    def forward(self, grad_output, x, approximate):
        return torch.ops.aten.gelu_backward(grad_output, x, approximate)

mod = Torch_Ops_Aten_GeluBackwardModule()

grad_output = torch.randn(3)
x = torch.randn(3)
approximate = torch.tensor(0)  # Fallback for unknown type str

args = (grad_output, x, approximate,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
