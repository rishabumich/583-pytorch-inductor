import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Stack_OutModule(torch.nn.Module):
    def forward(self, tensors, dim, out):
        return torch.ops.aten.stack.out(tensors, dim, out=out)

mod = Torch_Ops_Aten_Stack_OutModule()

tensors = torch.randn(3)
dim = 3
out = torch.empty(3)

args = (tensors, dim, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
