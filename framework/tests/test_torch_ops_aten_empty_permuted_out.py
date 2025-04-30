import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_EmptyPermuted_OutModule(torch.nn.Module):
    def forward(self, size, physical_layout, out):
        return torch.ops.aten.empty_permuted.out(size, physical_layout, out=out)

mod = Torch_Ops_Aten_EmptyPermuted_OutModule()

size = torch.sym_int(3)
physical_layout = 3
out = torch.empty(3)

args = (size, physical_layout, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
