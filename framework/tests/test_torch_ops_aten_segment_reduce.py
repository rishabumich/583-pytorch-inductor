import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SegmentReduceModule(torch.nn.Module):
    def forward(self, data, reduce, lengths, indices, offsets, axis, unsafe, initial):
        return torch.ops.aten.segment_reduce(data, reduce, lengths, indices, offsets, axis, unsafe, initial)

mod = Torch_Ops_Aten_SegmentReduceModule()

data = torch.randn(3)
reduce = None  # Fallback for unknown type str
lengths = torch.randn(3)
indices = torch.randn(3)
offsets = torch.randn(3)
axis = 3
unsafe = True
initial = 1

args = (data, reduce, lengths, indices, offsets, axis, unsafe, initial,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
