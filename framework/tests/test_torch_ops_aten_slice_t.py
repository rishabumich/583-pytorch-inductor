import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Slice_TModule(torch.nn.Module):
    def forward(self, l, start, end, step):
        return torch.ops.aten.slice.t(l, start, end, step)

mod = Torch_Ops_Aten_Slice_TModule()

l = torch.tensor(0)  # Fallback for unknown type |t[]
start = torch.tensor(0)  # Fallback for unknown type int?
end = torch.tensor(0)  # Fallback for unknown type int?
step = 3

args = (l, start, end, step,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
