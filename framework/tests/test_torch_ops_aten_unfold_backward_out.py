import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_UnfoldBackward_OutModule(torch.nn.Module):
    def forward(self, grad_in, input_sizes, dim, size, step, out):
        return torch.ops.aten.unfold_backward.out(grad_in, input_sizes, dim, size, step, out=out)

mod = Torch_Ops_Aten_UnfoldBackward_OutModule()

grad_in = torch.randn(3)
input_sizes = torch.sym_int(3)
dim = 3
size = 3
step = 3
out = torch.empty(3)

args = (grad_in, input_sizes, dim, size, step, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
