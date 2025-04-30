import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SliceBackward_OutModule(torch.nn.Module):
    def forward(self, grad_output, input_sizes, dim, start, end, step, out):
        return torch.ops.aten.slice_backward.out(grad_output, input_sizes, dim, start, end, step, out=out)

mod = Torch_Ops_Aten_SliceBackward_OutModule()

grad_output = torch.randn(3)
input_sizes = torch.sym_int(3)
dim = 3
start = None  # Fallback for unknown type SymInt
end = None  # Fallback for unknown type SymInt
step = None  # Fallback for unknown type SymInt
out = torch.empty(3)

args = (grad_output, input_sizes, dim, start, end, step, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
