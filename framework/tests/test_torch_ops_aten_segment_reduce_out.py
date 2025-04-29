import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_SegmentReduce_OutModule(torch.nn.Module):
    def forward(self, data, reduce, lengths, indices, offsets, axis, unsafe, initial, out):
        return torch.ops.aten.segment_reduce.out(data, reduce, lengths, indices, offsets, axis, unsafe, initial, out=out)

mod = Torch_Ops_Aten_SegmentReduce_OutModule()

data = torch.randn(3)
reduce = torch.tensor(0)  # Fallback for unknown type str
lengths = torch.randn(3)
indices = torch.randn(3)
offsets = torch.randn(3)
axis = 3
unsafe = True
initial = torch.tensor(0)  # Fallback for unknown type Scalar?
out = torch.empty(3)

args = (data, reduce, lengths, indices, offsets, axis, unsafe, initial, out,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
