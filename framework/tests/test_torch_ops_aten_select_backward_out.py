import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SelectBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, input_sizes, dim, index, out):
        return torch.ops.aten.select_backward.out(grad_output, input_sizes, dim, index, out=out)

mod = Torch_Ops_Aten_SelectBackward_OutModule()

grad_output = torch.randn(3)
input_sizes = torch.sym_int(3)
dim = 3
index = None  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (grad_output, input_sizes, dim, index, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
