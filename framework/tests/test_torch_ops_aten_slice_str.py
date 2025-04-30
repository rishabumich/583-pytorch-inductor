import torch
from torch.export import export_for_training
from torch._decomp import decomposition_table

class Torch_Ops_Aten_Slice_StrModule(torch.nn.Module):
    def forward(self, string, start, end, step):
        return torch.ops.aten.slice.str(string, start, end, step)

mod = Torch_Ops_Aten_Slice_StrModule()

string = None  # Fallback for unknown type |str
start = 3
end = 3
step = 3

args = (string, start, end, step,)

ep = export_for_training(mod, args)
print("Before decomposition:")
print(ep.module().code)

ep = ep.run_decompositions(decomposition_table)
print("After decomposition:")
print(ep.module().code)
