import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SliceBackwardModule(torch.nn.Module):
    def forward(self, grad_output, input_sizes, dim, start, end, step):
        return torch.ops.aten.slice_backward(grad_output, input_sizes, dim, start, end, step)

mod = Torch_Ops_Aten_SliceBackwardModule()

grad_output = torch.randn(3)
input_sizes = torch.sym_int(3)
dim = 3
start = None  # Fallback for unknown type SymInt
end = None  # Fallback for unknown type SymInt
step = None  # Fallback for unknown type SymInt

args = (grad_output, input_sizes, dim, start, end, step,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
