import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AsStridedScatter_OutModule(torch.nn.Module):
    def forward(self, x, src, size, stride, storage_offset, out):
        return torch.ops.aten.as_strided_scatter.out(x, src, size, stride, storage_offset, out=out)

mod = Torch_Ops_Aten_AsStridedScatter_OutModule()

x = torch.randn(3)
src = torch.randn(3)
size = torch.sym_int(3)
stride = torch.sym_int(3)
storage_offset = torch.sym_int(3)
out = torch.empty(3)

args = (x, src, size, stride, storage_offset, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
