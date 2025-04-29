import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AsStridedCopy_OutModule(torch.nn.Module):
    def forward(self, x, size, stride, storage_offset, out):
        return torch.ops.aten.as_strided_copy.out(x, size, stride, storage_offset, out=out)

mod = Torch_Ops_Aten_AsStridedCopy_OutModule()

x = torch.randn(3)
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
stride = torch.tensor(0)  # Fallback for unknown type SymInt[]
storage_offset = torch.tensor(0)  # Fallback for unknown type SymInt?
out = torch.empty(3)

args = (x, size, stride, storage_offset, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
