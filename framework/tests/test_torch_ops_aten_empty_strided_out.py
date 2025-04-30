import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EmptyStrided_OutModule(torch.nn.Module):
    def forward(self, size, stride, out):
        return torch.ops.aten.empty_strided.out(size, stride, out=out)

mod = Torch_Ops_Aten_EmptyStrided_OutModule()

size = torch.sym_int(3)
stride = torch.sym_int(3)
out = torch.empty(3)

args = (size, stride, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
