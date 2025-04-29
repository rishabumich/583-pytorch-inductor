import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_DiagonalBackwardModule(torch.nn.Module):
    def forward(self, grad_output, input_sizes, offset, dim1, dim2):
        return torch.ops.aten.diagonal_backward(grad_output, input_sizes, offset, dim1, dim2)

mod = Torch_Ops_Aten_DiagonalBackwardModule()

grad_output = torch.randn(3)
input_sizes = torch.tensor(0)  # Fallback for unknown type SymInt[]
offset = 3
dim1 = 3
dim2 = 3

args = (grad_output, input_sizes, offset, dim1, dim2,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
