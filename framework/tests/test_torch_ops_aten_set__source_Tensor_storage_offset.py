import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Set_SourceTensorStorageOffsetModule(torch.nn.Module):
    def forward(self, x, source, storage_offset, size, stride):
        return torch.ops.aten.set_.source_Tensor_storage_offset(x, source, storage_offset, size, stride)

mod = Torch_Ops_Aten_Set_SourceTensorStorageOffsetModule()

x = torch.randn(3)
source = torch.randn(3)
storage_offset = None  # Fallback for unknown type SymInt
size = torch.sym_int(3)
stride = torch.sym_int(3)

args = (x, source, storage_offset, size, stride,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
