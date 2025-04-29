import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Set_SourceStorageStorageOffsetModule(torch.nn.Module):
    def forward(self, x, source, storage_offset, size, stride):
        return torch.ops.aten.set_.source_Storage_storage_offset(x, source, storage_offset, size, stride)

mod = Torch_Ops_Aten_Set_SourceStorageStorageOffsetModule()

x = torch.randn(3)
source = torch.tensor(0)  # Fallback for unknown type Storage
storage_offset = torch.tensor(0)  # Fallback for unknown type SymInt
size = torch.tensor(0)  # Fallback for unknown type SymInt[]
stride = torch.tensor(0)  # Fallback for unknown type SymInt[]

args = (x, source, storage_offset, size, stride,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
