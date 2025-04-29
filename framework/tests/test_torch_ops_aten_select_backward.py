import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SelectBackwardModule(torch.nn.Module):
    def forward(self, grad_output, input_sizes, dim, index):
        return torch.ops.aten.select_backward(grad_output, input_sizes, dim, index)

mod = Torch_Ops_Aten_SelectBackwardModule()

grad_output = torch.randn(3)
input_sizes = torch.tensor(0)  # Fallback for unknown type SymInt[]
dim = 3
index = torch.tensor(0)  # Fallback for unknown type SymInt

args = (grad_output, input_sizes, dim, index,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
