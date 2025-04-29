import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_AsStridedModule(torch.nn.Module):
    def forward(self, x, size, stride, storage_offset):
        return torch.ops.aten.as_strided(x, size, stride, storage_offset)

mod = Torch_Ops_Aten_AsStridedModule()

x = torch.randn(3)
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
stride = torch.tensor(0)  # Fallback for unknown type SymInt[]
storage_offset = torch.tensor(0)  # Fallback for unknown type SymInt?

args = (x, size, stride, storage_offset,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
